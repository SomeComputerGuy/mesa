/*
 * Copyright © 2018 Red Hat
 * SPDX-License-Identifier: MIT
 */

#include "math.h"
#include "nir/nir_builtin_builder.h"

#include "util/u_printf.h"
#include "vtn_private.h"
#include "OpenCL.std.h"

typedef nir_def *(*nir_handler)(struct vtn_builder *b,
                                    uint32_t opcode,
                                    unsigned num_srcs, nir_def **srcs,
                                    struct vtn_type **src_types,
                                    const struct vtn_type *dest_type);

static int to_llvm_address_space(SpvStorageClass mode)
{
   switch (mode) {
   case SpvStorageClassPrivate:
   case SpvStorageClassFunction: return 0;
   case SpvStorageClassCrossWorkgroup: return 1;
   case SpvStorageClassUniform:
   case SpvStorageClassUniformConstant: return 2;
   case SpvStorageClassWorkgroup: return 3;
   case SpvStorageClassGeneric: return 4;
   default: return -1;
   }
}


static void
vtn_opencl_mangle(const char *in_name,
                  uint32_t const_mask,
                  int ntypes, struct vtn_type **src_types,
                  bool upcast_fp16,
                  char **outstring)
{
   char local_name[256] = "";
   char *args_str = local_name + sprintf(local_name, "_Z%zu%s", strlen(in_name), in_name);

   for (unsigned i = 0; i < ntypes; ++i) {
      const struct glsl_type *type = src_types[i]->type;
      enum vtn_base_type base_type = src_types[i]->base_type;
      if (src_types[i]->base_type == vtn_base_type_pointer) {
         *(args_str++) = 'P';
         int address_space = to_llvm_address_space(src_types[i]->storage_class);
         if (address_space > 0)
            args_str += sprintf(args_str, "U3AS%d", address_space);

         type = src_types[i]->pointed->type;
         base_type = src_types[i]->pointed->base_type;
      }

      if (const_mask & (1 << i))
         *(args_str++) = 'K';

      unsigned num_elements = glsl_get_components(type);
      if (num_elements > 1) {
         /* Vectors are not treated as built-ins for mangling, so check for substitution.
          * In theory, we'd need to know which substitution value this is. In practice,
          * the functions we need from libclc only support 1
          */
         bool substitution = false;
         for (unsigned j = 0; j < i; ++j) {
            const struct glsl_type *other_type = src_types[j]->base_type == vtn_base_type_pointer ?
               src_types[j]->pointed->type : src_types[j]->type;
            if (type == other_type) {
               substitution = true;
               break;
            }
         }

         if (substitution) {
            args_str += sprintf(args_str, "S_");
            continue;
         } else
            args_str += sprintf(args_str, "Dv%d_", num_elements);
      }

      const char *suffix = NULL;

      enum glsl_base_type glsl_base_type = glsl_get_base_type(type);
      if (glsl_base_type == GLSL_TYPE_FLOAT16 && upcast_fp16)
         glsl_base_type = GLSL_TYPE_FLOAT;

      switch (base_type) {
      case vtn_base_type_sampler: suffix = "11ocl_sampler"; break;
      case vtn_base_type_event: suffix = "9ocl_event"; break;
      default: {
         const char *primitives[] = {
            [GLSL_TYPE_UINT] = "j",
            [GLSL_TYPE_INT] = "i",
            [GLSL_TYPE_FLOAT] = "f",
            [GLSL_TYPE_FLOAT16] = "Dh",
            [GLSL_TYPE_DOUBLE] = "d",
            [GLSL_TYPE_UINT8] = "h",
            [GLSL_TYPE_INT8] = "c",
            [GLSL_TYPE_UINT16] = "t",
            [GLSL_TYPE_INT16] = "s",
            [GLSL_TYPE_UINT64] = "m",
            [GLSL_TYPE_INT64] = "l",
            [GLSL_TYPE_BOOL] = "b",
            [GLSL_TYPE_ERROR] = NULL,
         };
         assert(glsl_base_type < ARRAY_SIZE(primitives) && primitives[glsl_base_type]);
         suffix = primitives[glsl_base_type];
         break;
      }
      }
      args_str += sprintf(args_str, "%s", suffix);
   }

   *outstring = strdup(local_name);
}

static nir_function *mangle_and_find(struct vtn_builder *b,
                                     const char *name,
                                     uint8_t try_fp16_lowering,
                                     uint32_t const_mask,
                                     uint32_t num_srcs,
                                     struct vtn_type **src_types)
{
   char *mname;
   char *fp16_name = NULL;

   vtn_opencl_mangle(name, const_mask, num_srcs, src_types, false, &mname);

   /* try and find in current shader first. */
   nir_function *found = nir_shader_get_function_for_name(b->shader, mname);

   if (!found) {
      /* if not found here find in clc shader and create a decl mirroring it */
      if (b->options->clc_shader)
         found = nir_shader_get_function_for_name(b->options->clc_shader, mname);

      /* try upcasting fp16 */
      if (!found && try_fp16_lowering) {
         /* We might actually be inside libclc in which case clc_shader is NULL */
         const nir_shader *libclc = b->options->clc_shader ? b->options->clc_shader : b->shader;
         fp16_name = mname;
         vtn_opencl_mangle(name, const_mask, num_srcs, src_types, true, &mname);
         found = nir_shader_get_function_for_name(libclc, mname);
      }

      if (found) {
         nir_function *decl = nir_function_create(b->shader, mname);
         decl->num_params = found->num_params;
         decl->params = ralloc_array(b->shader, nir_parameter, decl->num_params);
         for (unsigned i = 0; i < decl->num_params; i++) {
            decl->params[i] = found->params[i];
            decl->params[i].name = ralloc_strdup(b->shader, found->params[i].name);
         }
         found = decl;

         if (fp16_name) {
            nir_function *fp16_decl = nir_function_create(b->shader, fp16_name);
            found = fp16_decl;

            fp16_decl->num_params = decl->num_params;
            fp16_decl->params = ralloc_array(b->shader, nir_parameter, fp16_decl->num_params);
            for (unsigned i = 0; i < fp16_decl->num_params; i++) {
               fp16_decl->params[i] = decl->params[i];
               if (try_fp16_lowering & (1 << i)) {
                  fp16_decl->params[i].type = glsl_f16vec_type(glsl_get_vector_elements(fp16_decl->params[i].type));
                  if (!fp16_decl->params[i].is_return) {
                     assert(fp16_decl->params[i].bit_size == 32);
                     fp16_decl->params[i].bit_size = 16;
                  }
               }
            }
            fp16_decl->impl = nir_function_impl_create(fp16_decl);

            nir_builder nb_saved = b->nb;
            b->nb = nir_builder_at(nir_before_impl(fp16_decl->impl));

            nir_variable *ret_tmp = nir_local_variable_create(b->nb.impl, glsl_get_bare_type(decl->params[0].type), "return_tmp");
            nir_deref_instr *ret_deref = nir_build_deref_var(&b->nb, ret_tmp);
            nir_call_instr *call = nir_call_instr_create(b->nb.shader, decl);

            call->params[0] = nir_src_for_ssa(&ret_deref->def);
            for (unsigned i = 1; i < fp16_decl->num_params; i++) {
               nir_def *param = nir_load_param(&b->nb, i);
               if (try_fp16_lowering & (1 << i))
                  param = nir_f2f32(&b->nb, param);
               call->params[i] = nir_src_for_ssa(param);
            }
            nir_builder_instr_insert(&b->nb, &call->instr);

            nir_def *res_val = nir_f2f16(&b->nb, nir_load_deref(&b->nb, ret_deref));
            nir_def *ret = nir_load_param(&b->nb, 0);
            ret_deref = nir_build_deref_cast(&b->nb, ret, nir_var_function_temp, fp16_decl->params[1].type, 0);
            nir_store_deref(&b->nb, ret_deref, res_val, -1);
            b->nb = nb_saved;
         }
      }
   }
   if (!found)
      vtn_fail("Can't find clc function %s\n", mname);
   free(mname);
   free(fp16_name);
   return found;
}

static bool call_mangled_function(struct vtn_builder *b,
                                  const char *name,
                                  uint8_t try_fp16_lowering,
                                  uint32_t const_mask,
                                  uint32_t num_srcs,
                                  struct vtn_type **src_types,
                                  const struct vtn_type *dest_type,
                                  nir_def **srcs,
                                  nir_deref_instr **ret_deref_ptr)
{
   nir_function *found = mangle_and_find(b, name, try_fp16_lowering, const_mask, num_srcs, src_types);
   if (!found)
      return false;

   nir_call_instr *call = nir_call_instr_create(b->shader, found);

   nir_deref_instr *ret_deref = NULL;
   uint32_t param_idx = 0;
   if (dest_type) {
      nir_variable *ret_tmp = nir_local_variable_create(b->nb.impl,
                                                        glsl_get_bare_type(dest_type->type),
                                                        "return_tmp");
      ret_deref = nir_build_deref_var(&b->nb, ret_tmp);
      call->params[param_idx++] = nir_src_for_ssa(&ret_deref->def);
   }

   for (unsigned i = 0; i < num_srcs; i++)
      call->params[param_idx++] = nir_src_for_ssa(srcs[i]);
   nir_builder_instr_insert(&b->nb, &call->instr);

   *ret_deref_ptr = ret_deref;
   return true;
}

static void
handle_instr(struct vtn_builder *b, uint32_t opcode,
             const uint32_t *w_src, unsigned num_srcs, const uint32_t *w_dest, nir_handler handler)
{
   struct vtn_type *dest_type = w_dest ? vtn_get_type(b, w_dest[0]) : NULL;

   nir_def *srcs[5] = { NULL };
   struct vtn_type *src_types[5] = { NULL };
   vtn_assert(num_srcs <= ARRAY_SIZE(srcs));
   for (unsigned i = 0; i < num_srcs; i++) {
      struct vtn_value *val = vtn_untyped_value(b, w_src[i]);
      struct vtn_ssa_value *ssa = vtn_ssa_value(b, w_src[i]);
      srcs[i] = ssa->def;
      src_types[i] = val->type;
   }

   nir_def *result = handler(b, opcode, num_srcs, srcs, src_types, dest_type);
   if (result) {
      vtn_push_nir_ssa(b, w_dest[1], result);
   } else {
      vtn_assert(dest_type == NULL);
   }
}

static void
handle_alu_instr(struct vtn_builder *b, uint32_t opcode,
                 const uint32_t *w_src, unsigned num_srcs, const uint32_t *w_dest, nir_handler handler)
{
   assert(w_dest);

   vtn_handle_fp_fast_math(b, vtn_untyped_value(b, w_dest[2]), vtn_untyped_value(b, w_src[0]));

   handle_instr(b, opcode, w_src, num_srcs, w_dest, handler);

   b->nb.fp_math_ctrl = nir_fp_fast_math;
}

static nir_op
nir_alu_op_for_opencl_opcode(struct vtn_builder *b,
                             enum OpenCLstd_Entrypoints opcode)
{
   switch (opcode) {
   case OpenCLstd_Fabs: return nir_op_fabs;
   case OpenCLstd_SAbs: return nir_op_iabs;
   case OpenCLstd_SAdd_sat: return nir_op_iadd_sat;
   case OpenCLstd_UAdd_sat: return nir_op_uadd_sat;
   case OpenCLstd_Ceil: return nir_op_fceil;
   case OpenCLstd_Floor: return nir_op_ffloor;
   case OpenCLstd_SHadd: return nir_op_ihadd;
   case OpenCLstd_UHadd: return nir_op_uhadd;
   case OpenCLstd_Fmax: return nir_op_fmax;
   case OpenCLstd_SMax: return nir_op_imax;
   case OpenCLstd_UMax: return nir_op_umax;
   case OpenCLstd_Fmin: return nir_op_fmin;
   case OpenCLstd_SMin: return nir_op_imin;
   case OpenCLstd_UMin: return nir_op_umin;
   case OpenCLstd_Mix: return nir_op_flrp;
   case OpenCLstd_Native_cos: return nir_op_fcos;
   case OpenCLstd_Native_divide: return nir_op_fdiv;
   case OpenCLstd_Native_exp2: return nir_op_fexp2;
   case OpenCLstd_Native_log2: return nir_op_flog2;
   case OpenCLstd_Native_powr: return nir_op_fpow;
   case OpenCLstd_Native_recip: return nir_op_frcp;
   case OpenCLstd_Native_rsqrt: return nir_op_frsq;
   case OpenCLstd_Native_sin: return nir_op_fsin;
   case OpenCLstd_Native_sqrt: return nir_op_fsqrt;
   case OpenCLstd_SMul_hi: return nir_op_imul_high;
   case OpenCLstd_UMul_hi: return nir_op_umul_high;
   case OpenCLstd_Popcount: return nir_op_bit_count;
   case OpenCLstd_SRhadd: return nir_op_irhadd;
   case OpenCLstd_URhadd: return nir_op_urhadd;
   case OpenCLstd_Rsqrt: return nir_op_frsq;
   case OpenCLstd_Sign: return nir_op_fsign;
   case OpenCLstd_Sqrt: return nir_op_fsqrt;
   case OpenCLstd_SSub_sat: return nir_op_isub_sat;
   case OpenCLstd_USub_sat: return nir_op_usub_sat;
   case OpenCLstd_Trunc: return nir_op_ftrunc;
   case OpenCLstd_Rint: return nir_op_fround_even;
   case OpenCLstd_Half_divide: return nir_op_fdiv;
   case OpenCLstd_Half_recip: return nir_op_frcp;
   /* uhm... */
   case OpenCLstd_UAbs: return nir_op_mov;
   // we could do better
   case OpenCLstd_FMin_common: return nir_op_fmin;
   case OpenCLstd_FMax_common: return nir_op_fmax;
   default:
      vtn_fail("No NIR equivalent");
   }
}

static nir_def *
handle_alu(struct vtn_builder *b, uint32_t opcode,
           unsigned num_srcs, nir_def **srcs, struct vtn_type **src_types,
           const struct vtn_type *dest_type)
{
   nir_def *ret = nir_build_alu(&b->nb, nir_alu_op_for_opencl_opcode(b, (enum OpenCLstd_Entrypoints)opcode),
                                    srcs[0], srcs[1], srcs[2], NULL);
   if (opcode == OpenCLstd_Popcount)
      ret = nir_u2uN(&b->nb, ret, glsl_get_bit_size(dest_type->type));
   return ret;
}

#define REMAP(op, str) [OpenCLstd_##op] = { str }
static const struct {
   const char *fn;
} remap_table[] = {
   REMAP(Distance, "distance"),
   REMAP(Fast_distance, "fast_distance"),
   REMAP(Fast_length, "fast_length"),
   REMAP(Fast_normalize, "fast_normalize"),
   REMAP(Half_rsqrt, "half_rsqrt"),
   REMAP(Half_sqrt, "half_sqrt"),
   REMAP(Length, "length"),
   REMAP(Normalize, "normalize"),
   REMAP(Degrees, "degrees"),
   REMAP(Radians, "radians"),
   REMAP(Rotate, "rotate"),
   REMAP(Smoothstep, "smoothstep"),
   REMAP(Step, "step"),

   REMAP(Pow, "pow"),
   REMAP(Pown, "pown"),
   REMAP(Powr, "powr"),
   REMAP(Rootn, "rootn"),
   REMAP(Modf, "modf"),

   REMAP(Acos, "acos"),
   REMAP(Acosh, "acosh"),
   REMAP(Acospi, "acospi"),
   REMAP(Asin, "asin"),
   REMAP(Asinh, "asinh"),
   REMAP(Asinpi, "asinpi"),
   REMAP(Atan, "atan"),
   REMAP(Atan2, "atan2"),
   REMAP(Atanh, "atanh"),
   REMAP(Atanpi, "atanpi"),
   REMAP(Atan2pi, "atan2pi"),
   REMAP(Cos, "cos"),
   REMAP(Cosh, "cosh"),
   REMAP(Cospi, "cospi"),
   REMAP(Sin, "sin"),
   REMAP(Sinh, "sinh"),
   REMAP(Sinpi, "sinpi"),
   REMAP(Tan, "tan"),
   REMAP(Tanh, "tanh"),
   REMAP(Tanpi, "tanpi"),
   REMAP(Sincos, "sincos"),
   REMAP(Fract, "fract"),
   REMAP(Frexp, "frexp"),
   REMAP(Fma, "fma"),
   REMAP(Fmod, "fmod"),

   REMAP(Half_cos, "cos"),
   REMAP(Half_exp, "exp"),
   REMAP(Half_exp2, "exp2"),
   REMAP(Half_exp10, "exp10"),
   REMAP(Half_log, "log"),
   REMAP(Half_log2, "log2"),
   REMAP(Half_log10, "log10"),
   REMAP(Half_powr, "powr"),
   REMAP(Half_sin, "sin"),
   REMAP(Half_tan, "tan"),

   REMAP(Remainder, "remainder"),
   REMAP(Remquo, "remquo"),
   REMAP(Hypot, "hypot"),
   REMAP(Exp, "exp"),
   REMAP(Exp2, "exp2"),
   REMAP(Exp10, "exp10"),
   REMAP(Expm1, "expm1"),
   REMAP(Ldexp, "ldexp"),

   REMAP(Ilogb, "ilogb"),
   REMAP(Log, "log"),
   REMAP(Log2, "log2"),
   REMAP(Log10, "log10"),
   REMAP(Log1p, "log1p"),
   REMAP(Logb, "logb"),

   REMAP(Cbrt, "cbrt"),
   REMAP(Erfc, "erfc"),
   REMAP(Erf, "erf"),

   REMAP(Lgamma, "lgamma"),
   REMAP(Lgamma_r, "lgamma_r"),
   REMAP(Tgamma, "tgamma"),

   REMAP(UMad_sat, "mad_sat"),
   REMAP(SMad_sat, "mad_sat"),

   REMAP(Shuffle, "shuffle"),
   REMAP(Shuffle2, "shuffle2"),
};
#undef REMAP

static const char *remap_clc_opcode(enum OpenCLstd_Entrypoints opcode)
{
   if (opcode >= (sizeof(remap_table) / sizeof(const char *)))
      return NULL;
   return remap_table[opcode].fn;
}

static struct vtn_type *
get_vtn_type_for_glsl_type(struct vtn_builder *b, const struct glsl_type *type)
{
   struct vtn_type *ret = vtn_zalloc(b, struct vtn_type);
   assert(glsl_type_is_vector_or_scalar(type));
   ret->type = type;
   ret->length = glsl_get_vector_elements(type);
   ret->base_type = glsl_type_is_vector(type) ? vtn_base_type_vector : vtn_base_type_scalar;
   return ret;
}

static struct vtn_type *
get_pointer_type(struct vtn_builder *b, struct vtn_type *t, SpvStorageClass storage_class)
{
   struct vtn_type *ret = vtn_zalloc(b, struct vtn_type);
   ret->type = nir_address_format_to_glsl_type(
            vtn_mode_to_address_format(
               b, vtn_storage_class_to_mode(b, storage_class, NULL, NULL)));
   ret->base_type = vtn_base_type_pointer;
   ret->storage_class = storage_class;
   ret->pointed = t;
   return ret;
}

static struct vtn_type *
get_signed_type(struct vtn_builder *b, struct vtn_type *t)
{
   if (t->base_type == vtn_base_type_pointer) {
      return get_pointer_type(b, get_signed_type(b, t->pointed), t->storage_class);
   }
   return get_vtn_type_for_glsl_type(
      b, glsl_vector_type(glsl_signed_base_type_of(glsl_get_base_type(t->type)),
                          glsl_get_vector_elements(t->type)));
}

static uint8_t fp16_lowering_supported(enum OpenCLstd_Entrypoints opcode)
{
   /* libclc has very limited fp16 compatibility */
   switch (opcode) {
   case OpenCLstd_Acos:
   case OpenCLstd_Acosh:
   case OpenCLstd_Acospi:
   case OpenCLstd_Asin:
   case OpenCLstd_Asinh:
   case OpenCLstd_Asinpi:
   case OpenCLstd_Atan:
   case OpenCLstd_Atan2:
   case OpenCLstd_Atanh:
   case OpenCLstd_Atanpi:
   case OpenCLstd_Atan2pi:
   case OpenCLstd_Cbrt:
   case OpenCLstd_Cos:
   case OpenCLstd_Cosh:
   case OpenCLstd_Cospi:
   case OpenCLstd_Degrees:
   case OpenCLstd_Distance:
   case OpenCLstd_Erf:
   case OpenCLstd_Erfc:
   case OpenCLstd_Exp:
   case OpenCLstd_Exp2:
   case OpenCLstd_Exp10:
   case OpenCLstd_Expm1:
   case OpenCLstd_Fma:
   case OpenCLstd_Fmod:
   case OpenCLstd_Fract:
   case OpenCLstd_Hypot:
   case OpenCLstd_Ilogb:
   case OpenCLstd_Length:
   case OpenCLstd_Lgamma:
   case OpenCLstd_Log:
   case OpenCLstd_Log2:
   case OpenCLstd_Log10:
   case OpenCLstd_Log1p:
   case OpenCLstd_Logb:
   case OpenCLstd_Modf:
   case OpenCLstd_Pow:
   case OpenCLstd_Powr:
   case OpenCLstd_Radians:
   case OpenCLstd_Remainder:
   case OpenCLstd_Smoothstep:
   case OpenCLstd_Step:
   case OpenCLstd_Sin:
   case OpenCLstd_Sinh:
   case OpenCLstd_Sinpi:
   case OpenCLstd_Tan:
   case OpenCLstd_Tanh:
   case OpenCLstd_Tanpi:
   case OpenCLstd_Tgamma:
      return 0xff;
   case OpenCLstd_Frexp:
   case OpenCLstd_Ldexp:
   case OpenCLstd_Lgamma_r:
   case OpenCLstd_Pown:
   case OpenCLstd_Rootn:
      /* second argument shouldn't be touched at all */
      return 0xff ^ (1 << 2);
   case OpenCLstd_Remquo:
      /* third argument is the integer quotient pointer. */
      return 0xff ^ (1 << 3);
   /* the second argument is a pointer to a float
    * a new enough libclc supports it though
    */
   case OpenCLstd_Sincos:
   default:
      return 0;
   }
}

static nir_def *
handle_clc_fn(struct vtn_builder *b, enum OpenCLstd_Entrypoints opcode,
              int num_srcs,
              nir_def **srcs,
              struct vtn_type **src_types,
              const struct vtn_type *dest_type)
{
   const char *name = remap_clc_opcode(opcode);
   if (!name)
       return NULL;

   /* Some functions which take params end up with uint (or pointer-to-uint) being passed,
    * which doesn't mangle correctly when the function expects int or pointer-to-int.
    * See https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_unsignedsigned_a_unsigned_versus_signed_integers
    */
   int signed_param = -1;
   switch (opcode) {
   case OpenCLstd_Frexp:
   case OpenCLstd_Lgamma_r:
   case OpenCLstd_Pown:
   case OpenCLstd_Rootn:
   case OpenCLstd_Ldexp:
      signed_param = 1;
      break;
   case OpenCLstd_Remquo:
      signed_param = 2;
      break;
   case OpenCLstd_SMad_sat: {
      /* All parameters need to be converted to signed */
      src_types[0] = src_types[1] = src_types[2] = get_signed_type(b, src_types[0]);
      break;
   }
   default: break;
   }

   if (signed_param >= 0) {
      src_types[signed_param] = get_signed_type(b, src_types[signed_param]);
   }

   nir_deref_instr *ret_deref = NULL;

   uint8_t try_fp16_lowering = fp16_lowering_supported(opcode);
   if (!call_mangled_function(b, name, try_fp16_lowering, 0, num_srcs, src_types,
                              dest_type, srcs, &ret_deref))
      return NULL;

   return ret_deref ? nir_load_deref(&b->nb, ret_deref) : NULL;
}

static nir_def *
handle_special(struct vtn_builder *b, uint32_t opcode,
               unsigned num_srcs, nir_def **srcs, struct vtn_type **src_types,
               const struct vtn_type *dest_type)
{
   nir_builder *nb = &b->nb;
   enum OpenCLstd_Entrypoints cl_opcode = (enum OpenCLstd_Entrypoints)opcode;

   switch (cl_opcode) {
   case OpenCLstd_SAbs_diff:
     /* these works easier in direct NIR */
      return nir_iabs_diff(nb, srcs[0], srcs[1]);
   case OpenCLstd_UAbs_diff:
      return nir_uabs_diff(nb, srcs[0], srcs[1]);
   case OpenCLstd_Bitselect:
      return nir_bitselect(nb, srcs[0], srcs[1], srcs[2]);
   case OpenCLstd_SMad_hi:
      return nir_imad_hi(nb, srcs[0], srcs[1], srcs[2]);
   case OpenCLstd_UMad_hi:
      return nir_umad_hi(nb, srcs[0], srcs[1], srcs[2]);
   case OpenCLstd_SMul24:
      return nir_imul24_relaxed(nb, srcs[0], srcs[1]);
   case OpenCLstd_UMul24:
      return nir_umul24_relaxed(nb, srcs[0], srcs[1]);
   case OpenCLstd_SMad24:
      return nir_iadd(nb, nir_imul24_relaxed(nb, srcs[0], srcs[1]), srcs[2]);
   case OpenCLstd_UMad24:
      return nir_umad24_relaxed(nb, srcs[0], srcs[1], srcs[2]);
   case OpenCLstd_FClamp:
      return nir_fclamp(nb, srcs[0], srcs[1], srcs[2]);
   case OpenCLstd_SClamp:
      return nir_iclamp(nb, srcs[0], srcs[1], srcs[2]);
   case OpenCLstd_UClamp:
      return nir_uclamp(nb, srcs[0], srcs[1], srcs[2]);
   case OpenCLstd_Copysign:
      return nir_copysign(nb, srcs[0], srcs[1]);
   case OpenCLstd_Cross:
      if (dest_type->length == 4)
         return nir_cross4(nb, srcs[0], srcs[1]);
      return nir_cross3(nb, srcs[0], srcs[1]);
   case OpenCLstd_Fdim:
      return nir_fdim(nb, srcs[0], srcs[1]);
   case OpenCLstd_Mad: {
      /* The spec says mad is
       *
       *    Implemented either as a correctly rounded fma or as a multiply
       *    followed by an add both of which are correctly rounded
       */

      const unsigned save_math_ctrl = nb->fp_math_ctrl;
      nb->fp_math_ctrl |= nir_fp_no_contract | nir_fp_no_transform;
      nir_def *res = nir_ffma_weak(nb, srcs[0], srcs[1], srcs[2]);
      nb->fp_math_ctrl = save_math_ctrl;
      return res;
   }
   case OpenCLstd_Maxmag:
      return nir_maxmag(nb, srcs[0], srcs[1]);
   case OpenCLstd_Minmag:
      return nir_minmag(nb, srcs[0], srcs[1]);
   case OpenCLstd_Nan:
      return nir_nan(nb, srcs[0]);
   case OpenCLstd_Nextafter:
      return nir_nextafter(nb, srcs[0], srcs[1]);
   case OpenCLstd_Normalize:
      return nir_normalize(nb, srcs[0]);
   case OpenCLstd_Clz:
      return nir_clz_u(nb, srcs[0]);
   case OpenCLstd_Ctz:
      return nir_ctz_u(nb, srcs[0]);
   case OpenCLstd_Select:
      return nir_select(nb, srcs[0], srcs[1], srcs[2]);
   case OpenCLstd_S_Upsample:
   case OpenCLstd_U_Upsample:
      /* SPIR-V and CL have different defs for upsample, just implement in nir */
      return nir_upsample(nb, srcs[0], srcs[1]);
   case OpenCLstd_Native_exp:
      return nir_fexp(nb, srcs[0]);
   case OpenCLstd_Native_exp10:
      return nir_fexp2(nb, nir_fmul_imm(nb, srcs[0], log(10) / log(2)));
   case OpenCLstd_Native_log:
      return nir_flog(nb, srcs[0]);
   case OpenCLstd_Native_log10:
      return nir_fmul_imm(nb, nir_flog2(nb, srcs[0]), log(2) / log(10));
   case OpenCLstd_Native_tan:
      return nir_ftan(nb, srcs[0]);
   case OpenCLstd_Ldexp:
      if (!nb->shader->options->has_ldexp)
         break;
      return nir_ldexp(nb, srcs[0], srcs[1]);
   case OpenCLstd_Fma: {
      /* FIXME: the software implementation only supports fp32 for now. */
      if (srcs[0]->bit_size != 64 && !nir_has_ffma(nb->shader, srcs[0]->bit_size))
         break;

      /* OpenCL FMA is not allowed to be split. */
      const unsigned save_math_ctrl = nb->fp_math_ctrl;
      nb->fp_math_ctrl |= nir_fp_exact;
      nir_def *res = nir_ffma(nb, srcs[0], srcs[1], srcs[2]);
      nb->fp_math_ctrl = save_math_ctrl;
      return res;
   }
   case OpenCLstd_Rotate:
      return nir_urol(nb, srcs[0], nir_u2u32(nb, srcs[1]));
   case OpenCLstd_Log2:
   case OpenCLstd_Log:
   case OpenCLstd_Exp2:
      /* Emit precise log2/log/exp2 as hardware transcendentals when the
       * driver's hardware units meet the CL ULP budget. fp32 only; exp/pow
       * are intentionally excluded (change-of-base exceeds ULP budget). */
      if (b->shader->options->has_cl_log2_exp2 && srcs[0]->bit_size == 32) {
         switch (cl_opcode) {
         case OpenCLstd_Log2:
            return nir_flog2(nb, srcs[0]);
         case OpenCLstd_Log:
            return nir_fmul_imm(nb, nir_flog2(nb, srcs[0]), log(2.0));
         case OpenCLstd_Exp2:
            return nir_fexp2(nb, srcs[0]);
         default:
            break;
         }
      }
      break;
   default:
      break;
   }

   nir_def *ret = handle_clc_fn(b, opcode, num_srcs, srcs, src_types, dest_type);
   if (!ret)
      vtn_fail("No NIR equivalent");

   switch (opcode) {
   /* libclc's cbrt() implementation fails to flush subnormal numbers to zero
    * even when flush-to-zero is required. Manually flush its output.
    */
   case OpenCLstd_Cbrt:
      ret = nir_fcanonicalize(nb, ret);
      break;

   /* Cospi is always expected to return +0.0 instead of -0.0 */
   case OpenCLstd_Cospi: {
      if (nb->fp_math_ctrl & nir_fp_preserve_signed_zero)
         ret = nir_fadd_imm(nb, ret, 0.0);
      break;
   }

   /* Sinpi expects a resulting zero to be of the same sign as the input */
   case OpenCLstd_Sinpi: {
      if (nb->fp_math_ctrl & nir_fp_preserve_signed_zero) {
         ret = nir_bcsel(nb, nir_feq_imm(nb, ret, 0.0), nir_copysign(nb, ret, srcs[0]), ret);
      }
      break;
   }

   case OpenCLstd_Tanpi: {
      if (nb->fp_math_ctrl & nir_fp_preserve_signed_zero) {
         nir_def *remainder = nir_fmod(nb, nir_fabs(nb, srcs[0]), nir_imm_floatN_t(nb, 2.0, ret->bit_size));
         nir_def *is_odd = nir_feq_imm(nb, remainder, 1.0);
         nir_def *is_even = nir_feq_imm(nb, remainder, 0.0);

         /* tanpi(n) is copysign(0.0, - n) for odd integers n. */
         ret = nir_bcsel(nb, is_odd, nir_copysign(nb, ret, nir_fneg(nb, srcs[0])), ret);

         /* tanpi(n) is copysign(0.0, n) for even integers n. */
         ret = nir_bcsel(nb, is_even, nir_copysign(nb, ret, srcs[0]), ret);
      }
      break;
   }

   default:
      break;
   }

   return ret;
}

static nir_def *
handle_core(struct vtn_builder *b, uint32_t opcode,
            unsigned num_srcs, nir_def **srcs, struct vtn_type **src_types,
            const struct vtn_type *dest_type)
{
   nir_deref_instr *ret_deref = NULL;

   switch ((SpvOp)opcode) {
   case SpvOpGroupAsyncCopy: {
      /* Libclc doesn't include 3-component overloads of the async copy functions.
       * However, the CLC spec says:
       * async_work_group_copy and async_work_group_strided_copy for 3-component vector types
       * behave as async_work_group_copy and async_work_group_strided_copy respectively for 4-component
       * vector types
       */
      for (unsigned i = 0; i < num_srcs; ++i) {
         if (src_types[i]->base_type == vtn_base_type_pointer &&
             src_types[i]->pointed->base_type == vtn_base_type_vector &&
             src_types[i]->pointed->length == 3) {
            src_types[i] =
               get_pointer_type(b,
                                get_vtn_type_for_glsl_type(b, glsl_replace_vector_type(src_types[i]->pointed->type, 4)),
                                src_types[i]->storage_class);
         }
      }
      if (!call_mangled_function(b, "async_work_group_strided_copy", false, (1 << 1), num_srcs, src_types, dest_type, srcs, &ret_deref))
         return NULL;
      break;
   }
   case SpvOpGroupWaitEvents: {
      /* libclc and clang don't agree on the mangling of this function.
       * The libclc we have uses a __local pointer but clang gives us generic
       * pointers.  Fortunately, the whole function is just a barrier.
       */
      nir_barrier(&b->nb, .execution_scope = SCOPE_WORKGROUP,
                          .memory_scope = SCOPE_DEVICE,
                          .memory_semantics = NIR_MEMORY_ACQUIRE |
                                              NIR_MEMORY_RELEASE,
                          .memory_modes = nir_var_mem_shared |
                                          nir_var_mem_global);
      break;
   }
   default:
      return NULL;
   }

   return ret_deref ? nir_load_deref(&b->nb, ret_deref) : NULL;
}


static void
_handle_v_load_store(struct vtn_builder *b, enum OpenCLstd_Entrypoints opcode,
                     const uint32_t *w, unsigned count, bool load,
                     bool vec_aligned, nir_rounding_mode rounding)
{
   struct vtn_type *type;
   if (load)
      type = vtn_get_type(b, w[1]);
   else
      type = vtn_get_value_type(b, w[5]);
   unsigned a = load ? 0 : 1;

   enum glsl_base_type base_type = glsl_get_base_type(type->type);
   unsigned components = glsl_get_vector_elements(type->type);

   nir_def *offset = vtn_get_nir_ssa(b, w[5 + a]);
   struct vtn_value *p = vtn_value(b, w[6 + a], vtn_value_type_pointer);

   struct vtn_ssa_value *comps[NIR_MAX_VEC_COMPONENTS];
   nir_def *ncomps[NIR_MAX_VEC_COMPONENTS];

   nir_def *moffset = nir_imul_imm(&b->nb, offset,
      (vec_aligned && components == 3) ? 4 : components);
   nir_deref_instr *deref = vtn_pointer_to_deref(b, p->pointer);

   unsigned alignment = vec_aligned ? glsl_get_cl_alignment(type->type) :
                                      glsl_get_bit_size(type->type) / 8;
   enum glsl_base_type ptr_base_type =
      glsl_get_base_type(p->pointer->type->pointed->type);
   if (base_type != ptr_base_type) {
      vtn_fail_if(ptr_base_type != GLSL_TYPE_FLOAT16 ||
                  (base_type != GLSL_TYPE_FLOAT &&
                   base_type != GLSL_TYPE_DOUBLE),
                  "vload/vstore cannot do type conversion. "
                  "vload/vstore_half can only convert from half to other "
                  "floating-point types.");

      /* Above-computed alignment was for floats/doubles, not halves */
      alignment /= glsl_get_bit_size(type->type) / glsl_base_type_get_bit_size(ptr_base_type);
   }

   deref = nir_alignment_deref_cast(&b->nb, deref, alignment, 0);

   for (int i = 0; i < components; i++) {
      nir_def *coffset = nir_iadd_imm(&b->nb, moffset, i);
      nir_deref_instr *arr_deref = nir_build_deref_ptr_as_array(&b->nb, deref, coffset);

      if (load) {
         comps[i] = vtn_local_load(b, arr_deref, p->type->access);
         ncomps[i] = comps[i]->def;
         if (base_type != ptr_base_type) {
            assert(ptr_base_type == GLSL_TYPE_FLOAT16 &&
                   (base_type == GLSL_TYPE_FLOAT ||
                    base_type == GLSL_TYPE_DOUBLE));
            ncomps[i] = nir_f2fN(&b->nb, ncomps[i],
                                 glsl_base_type_get_bit_size(base_type));
         }
      } else {
         struct vtn_ssa_value *ssa = vtn_create_ssa_value(b, glsl_scalar_type(base_type));
         struct vtn_ssa_value *val = vtn_ssa_value(b, w[5]);
         ssa->def = nir_channel(&b->nb, val->def, i);
         if (base_type != ptr_base_type) {
            assert(ptr_base_type == GLSL_TYPE_FLOAT16 &&
                   (base_type == GLSL_TYPE_FLOAT ||
                    base_type == GLSL_TYPE_DOUBLE));
            if (rounding == nir_rounding_mode_undef) {
               ssa->def = nir_f2f16(&b->nb, ssa->def);
            } else {
               ssa->def = nir_convert_alu_types(&b->nb, 16, ssa->def,
                                                nir_type_float | ssa->def->bit_size,
                                                nir_type_float16,
                                                rounding, false);
            }
         }
         vtn_local_store(b, ssa, arr_deref, p->type->access);
      }
   }
   if (load) {
      vtn_push_nir_ssa(b, w[2], nir_vec(&b->nb, ncomps, components));
   }
}

static void
vtn_handle_opencl_vload(struct vtn_builder *b, enum OpenCLstd_Entrypoints opcode,
                        const uint32_t *w, unsigned count)
{
   _handle_v_load_store(b, opcode, w, count, true,
                        opcode == OpenCLstd_Vloada_halfn,
                        nir_rounding_mode_undef);
}

static void
vtn_handle_opencl_vstore(struct vtn_builder *b, enum OpenCLstd_Entrypoints opcode,
                         const uint32_t *w, unsigned count)
{
   _handle_v_load_store(b, opcode, w, count, false,
                        opcode == OpenCLstd_Vstorea_halfn,
                        nir_rounding_mode_undef);
}

static void
vtn_handle_opencl_vstore_half_r(struct vtn_builder *b, enum OpenCLstd_Entrypoints opcode,
                                const uint32_t *w, unsigned count)
{
   _handle_v_load_store(b, opcode, w, count, false,
                        opcode == OpenCLstd_Vstorea_halfn_r,
                        vtn_rounding_mode_to_nir(b, w[8]));
}

static unsigned
vtn_add_printf_string(struct vtn_builder *b, uint32_t id, u_printf_info *info)
{
   nir_deref_instr *deref = vtn_nir_deref(b, id);

   while (deref->deref_type != nir_deref_type_var) {
      nir_scalar parent = nir_scalar_resolved(deref->parent.ssa, 0);
      if (!nir_def_is_deref(parent.def)) {
         deref = NULL;
         break;
      }
      vtn_assert(parent.comp == 0);
      deref = nir_def_as_deref(parent.def);
   }

   vtn_fail_if(deref == NULL || !nir_deref_mode_is(deref, nir_var_mem_constant),
               "Printf string argument must be a pointer to a constant variable");
   vtn_fail_if(deref->var->constant_initializer == NULL,
               "Printf string argument must have an initializer");
   vtn_fail_if(!glsl_type_is_array(deref->var->type),
               "Printf string must be an char array");
   const struct glsl_type *char_type = glsl_get_array_element(deref->var->type);
   vtn_fail_if(char_type != glsl_uint8_t_type() &&
               char_type != glsl_int8_t_type(),
               "Printf string must be an char array");

   nir_constant *c = deref->var->constant_initializer;
   assert(c->num_elements == glsl_get_length(deref->var->type));

   unsigned idx = info->string_size;
   info->strings = reralloc_size(b->shader, info->strings,
                                 idx + c->num_elements);
   info->string_size += c->num_elements;

   char *str = &info->strings[idx];
   bool found_null = false;
   for (unsigned i = 0; i < c->num_elements; i++) {
      memcpy((char *)str + i, c->elements[i]->values, 1);
      if (str[i] == '\0')
         found_null = true;
   }
   vtn_fail_if(!found_null, "Printf string must be null terminated");
   return idx;
}

/* printf is special because there are no limits on args */
static void
handle_printf(struct vtn_builder *b, uint32_t opcode,
              const uint32_t *w_src, unsigned num_srcs, const uint32_t *w_dest)
{
   if (!b->options->printf) {
      vtn_push_nir_ssa(b, w_dest[1], nir_imm_int(&b->nb, -1));
      return;
   }

   /* Step 1. extract the format string */

   /*
    * info_idx is 1-based to match clover/llvm
    * the backend indexes the info table at info_idx - 1.
    */
   b->shader->printf_info_count++;
   unsigned info_idx = b->shader->printf_info_count;

   b->shader->printf_info = reralloc(b->shader, b->shader->printf_info,
                                     u_printf_info, info_idx);
   u_printf_info *info = &b->shader->printf_info[info_idx - 1];

   info->strings = NULL;
   info->string_size = 0;

   vtn_add_printf_string(b, w_src[0], info);

   info->num_args = num_srcs - 1;
   info->arg_sizes = ralloc_array(b->shader, unsigned, info->num_args);

   /* Step 2, build an ad-hoc struct type out of the args */
   unsigned field_offset = 0;
   struct glsl_struct_field *fields =
      rzalloc_array(b, struct glsl_struct_field, num_srcs - 1);
   for (unsigned i = 1; i < num_srcs; ++i) {
      struct vtn_value *val = vtn_untyped_value(b, w_src[i]);
      struct vtn_type *src_type = val->type;
      fields[i - 1].type = src_type->type;
      fields[i - 1].name = ralloc_asprintf(b->shader, "arg_%u", i);
      field_offset = align(field_offset, 4);
      fields[i - 1].offset = field_offset;
      info->arg_sizes[i - 1] = glsl_get_cl_size(src_type->type);
      field_offset += glsl_get_cl_size(src_type->type);
   }
   const struct glsl_type *struct_type =
      glsl_struct_type(fields, num_srcs - 1, "printf", true);

   /* Step 3, create a variable of that type and populate its fields */
   nir_variable *var = nir_local_variable_create(b->nb.impl, struct_type, NULL);
   nir_deref_instr *deref_var = nir_build_deref_var(&b->nb, var);
   size_t fmt_pos = 0;
   for (unsigned i = 1; i < num_srcs; ++i) {
      nir_deref_instr *field_deref =
         nir_build_deref_struct(&b->nb, deref_var, i - 1);
      nir_def *field_src = vtn_ssa_value(b, w_src[i])->def;
      /* extract strings */
      fmt_pos = util_printf_next_spec_pos(info->strings, fmt_pos);
      if (fmt_pos != -1 && info->strings[fmt_pos] == 's') {
         unsigned idx = vtn_add_printf_string(b, w_src[i], info);
         nir_store_deref(&b->nb, field_deref,
                         nir_imm_intN_t(&b->nb, idx, field_src->bit_size),
                         ~0 /* write_mask */);
      } else
         nir_store_deref(&b->nb, field_deref, field_src, ~0);
   }

   /* Lastly, the actual intrinsic */
   nir_def *ret = nir_printf(&b->nb, &deref_var->def, .fmt_idx = info_idx);
   vtn_push_nir_ssa(b, w_dest[1], ret);

   b->nb.shader->info.uses_printf = true;
}

static nir_def *
handle_round(struct vtn_builder *b, uint32_t opcode,
             unsigned num_srcs, nir_def **srcs, struct vtn_type **src_types,
             const struct vtn_type *dest_type)
{
   nir_def *src = srcs[0];
   nir_builder *nb = &b->nb;
   nir_def *half = nir_imm_floatN_t(nb, 0.5, src->bit_size);
   nir_def *truncated = nir_ftrunc(nb, src);
   nir_def *remainder = nir_fsub(nb, src, truncated);

   return nir_bcsel(nb, nir_fge(nb, nir_fabs(nb, remainder), half),
                    nir_fadd(nb, truncated, nir_fsign(nb, src)), truncated);
}

static nir_def *
handle_shuffle(struct vtn_builder *b, uint32_t opcode,
               unsigned num_srcs, nir_def **srcs, struct vtn_type **src_types,
               const struct vtn_type *dest_type)
{
   struct nir_def *input = srcs[0];
   struct nir_def *mask = srcs[1];

   unsigned out_elems = dest_type->length;
   nir_def *outres[NIR_MAX_VEC_COMPONENTS];
   unsigned in_elems = input->num_components;
   if (mask->bit_size != 32)
      mask = nir_u2u32(&b->nb, mask);
   mask = nir_iand(&b->nb, mask, nir_imm_intN_t(&b->nb, in_elems - 1, mask->bit_size));
   for (unsigned i = 0; i < out_elems; i++)
      outres[i] = nir_vector_extract(&b->nb, input, nir_channel(&b->nb, mask, i));

   return nir_vec(&b->nb, outres, out_elems);
}

static nir_def *
handle_shuffle2(struct vtn_builder *b, uint32_t opcode,
                unsigned num_srcs, nir_def **srcs, struct vtn_type **src_types,
                const struct vtn_type *dest_type)
{
   struct nir_def *input0 = srcs[0];
   struct nir_def *input1 = srcs[1];
   struct nir_def *mask = srcs[2];

   unsigned out_elems = dest_type->length;
   nir_def *outres[NIR_MAX_VEC_COMPONENTS];
   unsigned in_elems = input0->num_components;
   unsigned total_mask = 2 * in_elems - 1;
   unsigned half_mask = in_elems - 1;
   if (mask->bit_size != 32)
      mask = nir_u2u32(&b->nb, mask);
   mask = nir_iand(&b->nb, mask, nir_imm_intN_t(&b->nb, total_mask, mask->bit_size));
   for (unsigned i = 0; i < out_elems; i++) {
      nir_def *this_mask = nir_channel(&b->nb, mask, i);
      nir_def *vmask = nir_iand(&b->nb, this_mask, nir_imm_intN_t(&b->nb, half_mask, mask->bit_size));
      nir_def *val0 = nir_vector_extract(&b->nb, input0, vmask);
      nir_def *val1 = nir_vector_extract(&b->nb, input1, vmask);
      nir_def *sel = nir_ilt_imm(&b->nb, this_mask, in_elems);
      outres[i] = nir_bcsel(&b->nb, sel, val0, val1);
   }
   return nir_vec(&b->nb, outres, out_elems);
}

bool
vtn_handle_opencl_instruction(struct vtn_builder *b, SpvOp ext_opcode,
                              const uint32_t *w, unsigned count)
{
   enum OpenCLstd_Entrypoints cl_opcode = (enum OpenCLstd_Entrypoints) ext_opcode;

   switch (cl_opcode) {
   case OpenCLstd_Fabs:
   case OpenCLstd_SAbs:
   case OpenCLstd_UAbs:
   case OpenCLstd_SAdd_sat:
   case OpenCLstd_UAdd_sat:
   case OpenCLstd_Ceil:
   case OpenCLstd_Floor:
   case OpenCLstd_Fmax:
   case OpenCLstd_SHadd:
   case OpenCLstd_UHadd:
   case OpenCLstd_SMax:
   case OpenCLstd_UMax:
   case OpenCLstd_Fmin:
   case OpenCLstd_SMin:
   case OpenCLstd_UMin:
   case OpenCLstd_Mix:
   case OpenCLstd_Native_cos:
   case OpenCLstd_Native_divide:
   case OpenCLstd_Native_exp2:
   case OpenCLstd_Native_log2:
   case OpenCLstd_Native_powr:
   case OpenCLstd_Native_recip:
   case OpenCLstd_Native_rsqrt:
   case OpenCLstd_Native_sin:
   case OpenCLstd_Native_sqrt:
   case OpenCLstd_SMul_hi:
   case OpenCLstd_UMul_hi:
   case OpenCLstd_Popcount:
   case OpenCLstd_SRhadd:
   case OpenCLstd_URhadd:
   case OpenCLstd_Rsqrt:
   case OpenCLstd_Sign:
   case OpenCLstd_Sqrt:
   case OpenCLstd_SSub_sat:
   case OpenCLstd_USub_sat:
   case OpenCLstd_Trunc:
   case OpenCLstd_Rint:
   case OpenCLstd_Half_divide:
   case OpenCLstd_Half_recip:
   case OpenCLstd_FMin_common:
   case OpenCLstd_FMax_common:
      handle_alu_instr(b, ext_opcode, w + 5, count - 5, w + 1, handle_alu);
      return true;
   case OpenCLstd_SAbs_diff:
   case OpenCLstd_UAbs_diff:
   case OpenCLstd_SMad_hi:
   case OpenCLstd_UMad_hi:
   case OpenCLstd_SMad24:
   case OpenCLstd_UMad24:
   case OpenCLstd_SMul24:
   case OpenCLstd_UMul24:
   case OpenCLstd_Bitselect:
   case OpenCLstd_FClamp:
   case OpenCLstd_SClamp:
   case OpenCLstd_UClamp:
   case OpenCLstd_Copysign:
   case OpenCLstd_Cross:
   case OpenCLstd_Degrees:
   case OpenCLstd_Fdim:
   case OpenCLstd_Fma:
   case OpenCLstd_Distance:
   case OpenCLstd_Fast_distance:
   case OpenCLstd_Fast_length:
   case OpenCLstd_Fast_normalize:
   case OpenCLstd_Half_rsqrt:
   case OpenCLstd_Half_sqrt:
   case OpenCLstd_Length:
   case OpenCLstd_Mad:
   case OpenCLstd_Maxmag:
   case OpenCLstd_Minmag:
   case OpenCLstd_Nan:
   case OpenCLstd_Nextafter:
   case OpenCLstd_Normalize:
   case OpenCLstd_Radians:
   case OpenCLstd_Rotate:
   case OpenCLstd_Select:
   case OpenCLstd_Step:
   case OpenCLstd_Smoothstep:
   case OpenCLstd_S_Upsample:
   case OpenCLstd_U_Upsample:
   case OpenCLstd_Clz:
   case OpenCLstd_Ctz:
   case OpenCLstd_Native_exp:
   case OpenCLstd_Native_exp10:
   case OpenCLstd_Native_log:
   case OpenCLstd_Native_log10:
   case OpenCLstd_Acos:
   case OpenCLstd_Acosh:
   case OpenCLstd_Acospi:
   case OpenCLstd_Asin:
   case OpenCLstd_Asinh:
   case OpenCLstd_Asinpi:
   case OpenCLstd_Atan:
   case OpenCLstd_Atan2:
   case OpenCLstd_Atanh:
   case OpenCLstd_Atanpi:
   case OpenCLstd_Atan2pi:
   case OpenCLstd_Fract:
   case OpenCLstd_Frexp:
   case OpenCLstd_Exp:
   case OpenCLstd_Exp2:
   case OpenCLstd_Expm1:
   case OpenCLstd_Exp10:
   case OpenCLstd_Fmod:
   case OpenCLstd_Ilogb:
   case OpenCLstd_Log:
   case OpenCLstd_Log2:
   case OpenCLstd_Log10:
   case OpenCLstd_Log1p:
   case OpenCLstd_Logb:
   case OpenCLstd_Ldexp:
   case OpenCLstd_Cos:
   case OpenCLstd_Cosh:
   case OpenCLstd_Cospi:
   case OpenCLstd_Sin:
   case OpenCLstd_Sinh:
   case OpenCLstd_Sinpi:
   case OpenCLstd_Tan:
   case OpenCLstd_Tanh:
   case OpenCLstd_Tanpi:
   case OpenCLstd_Cbrt:
   case OpenCLstd_Erfc:
   case OpenCLstd_Erf:
   case OpenCLstd_Lgamma:
   case OpenCLstd_Lgamma_r:
   case OpenCLstd_Tgamma:
   case OpenCLstd_Pow:
   case OpenCLstd_Powr:
   case OpenCLstd_Pown:
   case OpenCLstd_Rootn:
   case OpenCLstd_Remainder:
   case OpenCLstd_Remquo:
   case OpenCLstd_Hypot:
   case OpenCLstd_Sincos:
   case OpenCLstd_Modf:
   case OpenCLstd_UMad_sat:
   case OpenCLstd_SMad_sat:
   case OpenCLstd_Native_tan:
   case OpenCLstd_Half_cos:
   case OpenCLstd_Half_exp:
   case OpenCLstd_Half_exp2:
   case OpenCLstd_Half_exp10:
   case OpenCLstd_Half_log:
   case OpenCLstd_Half_log2:
   case OpenCLstd_Half_log10:
   case OpenCLstd_Half_powr:
   case OpenCLstd_Half_sin:
   case OpenCLstd_Half_tan:
      handle_alu_instr(b, ext_opcode, w + 5, count - 5, w + 1, handle_special);
      return true;
   case OpenCLstd_Vloadn:
   case OpenCLstd_Vload_half:
   case OpenCLstd_Vload_halfn:
   case OpenCLstd_Vloada_halfn:
      vtn_handle_opencl_vload(b, cl_opcode, w, count);
      return true;
   case OpenCLstd_Vstoren:
   case OpenCLstd_Vstore_half:
   case OpenCLstd_Vstore_halfn:
   case OpenCLstd_Vstorea_halfn:
      vtn_handle_opencl_vstore(b, cl_opcode, w, count);
      return true;
   case OpenCLstd_Vstore_half_r:
   case OpenCLstd_Vstore_halfn_r:
   case OpenCLstd_Vstorea_halfn_r:
      vtn_handle_opencl_vstore_half_r(b, cl_opcode, w, count);
      return true;
   case OpenCLstd_Shuffle:
      handle_instr(b, ext_opcode, w + 5, count - 5, w + 1, handle_shuffle);
      return true;
   case OpenCLstd_Shuffle2:
      handle_instr(b, ext_opcode, w + 5, count - 5, w + 1, handle_shuffle2);
      return true;
   case OpenCLstd_Round:
      handle_instr(b, ext_opcode, w + 5, count - 5, w + 1, handle_round);
      return true;
   case OpenCLstd_Printf:
      handle_printf(b, ext_opcode, w + 5, count - 5, w + 1);
      return true;
   case OpenCLstd_Prefetch:
      /* TODO maybe add a nir instruction for this? */
      return true;
   default:
      vtn_fail("unhandled opencl opc: %u\n", ext_opcode);
      return false;
   }
}

bool
vtn_handle_opencl_core_instruction(struct vtn_builder *b, SpvOp opcode,
                                   const uint32_t *w, unsigned count)
{
   switch (opcode) {
   case SpvOpGroupAsyncCopy:
      handle_instr(b, opcode, w + 4, count - 4, w + 1, handle_core);
      return true;
   case SpvOpGroupWaitEvents:
      handle_instr(b, opcode, w + 2, count - 2, NULL, handle_core);
      return true;
   default:
      return false;
   }
   return true;
}

/* OpenCL 2.0 pipes. A pipe is the global address of a buffer:
 *
 *   uint32 write_index, read_index, max_packets, packet_size;
 *   packet ring[max_packets];
 *   packet spare;    (target of failed writes)
 *
 * The indices only grow; packet i lives at ring[i % max_packets]. A kernel
 * only reads or only writes a given pipe, and kernels on a queue do not
 * overlap here, so reserving is a compare-and-swap on one index against the
 * other one (nir_intrinsic_pipe_reserve_*, lowered by the frontend after
 * structurizing) and commits have nothing to do. Copying packets needs no
 * control flow: a failed write goes to the spare packet, a failed read keeps
 * the old contents of the destination.
 */
#define VTN_PIPE_HEADER 16

static nir_def *
vtn_pipe_load_u32(nir_builder *b, nir_def *pipe, unsigned offset)
{
   return nir_load_global(b, 1, 32, nir_iadd_imm(b, pipe, offset),
                          .align_mul = 4,
                          .access = ACCESS_COHERENT | ACCESS_VOLATILE);
}

static void
vtn_pipe_copy(struct vtn_builder *b, nir_def *pipe, nir_def *index,
              nir_def *ok, uint32_t ptr_id, unsigned size, unsigned align,
              bool write)
{
   nir_builder *nb = &b->nb;
   nir_def *max = vtn_pipe_load_u32(nb, pipe, 8);
   nir_def *slot = nir_imul_imm(nb, nir_umod(nb, index, max), size);
   if (write)
      slot = nir_bcsel(nb, ok, slot, nir_imul_imm(nb, max, size));
   nir_def *addr = nir_iadd(nb, nir_iadd_imm(nb, pipe, VTN_PIPE_HEADER),
                            nir_u2u64(nb, slot));

   /* at most 16 chunks of a vector packet (nir_opt_deref turns these loads
    * of a vector into a vector load + bitcast): up to 8 bytes
    */
   unsigned chunk = 8;
   while (chunk > 1 && (size % chunk || align % chunk))
      chunk /= 2;
   const struct glsl_type *type = glsl_uintN_t_type(chunk * 8);

   struct vtn_pointer *ptr = vtn_value(b, ptr_id, vtn_value_type_pointer)->pointer;
   nir_deref_instr *user = vtn_pointer_to_deref(b, ptr);
   nir_deref_instr *user_cast =
      nir_build_deref_cast_with_alignment(nb, &user->def, user->modes, type,
                                          chunk, chunk, 0);
   nir_deref_instr *ring_cast =
      nir_build_deref_cast_with_alignment(nb, addr, nir_var_mem_global, type,
                                          chunk, chunk, 0);

   for (unsigned i = 0; i < size / chunk; i++) {
      nir_deref_instr *u = nir_build_deref_ptr_as_array(nb, user_cast, nir_imm_int64(nb, i));
      nir_deref_instr *r = nir_build_deref_ptr_as_array(nb, ring_cast, nir_imm_int64(nb, i));
      if (write) {
         nir_store_deref(nb, r, nir_load_deref(nb, u), 1);
      } else {
         nir_def *v = nir_load_deref(nb, r);
         nir_store_deref(nb, u, nir_bcsel(nb, ok, v, nir_load_deref(nb, u)), 1);
      }
   }
}

static nir_def *
vtn_pipe_reserve(struct vtn_builder *b, nir_def *pipe, nir_def *num,
                 mesa_scope scope, bool write)
{
   nir_builder *nb = &b->nb;
   num = nir_u2u32(nb, num);
   return write ? nir_pipe_reserve_write(nb, pipe, num, .execution_scope = scope)
                : nir_pipe_reserve_read(nb, pipe, num, .execution_scope = scope);
}

void
vtn_handle_pipe(struct vtn_builder *b, SpvOp opcode,
                const uint32_t *w, unsigned count)
{
   nir_builder *nb = &b->nb;
   nir_def *res = NULL;

   switch (opcode) {
   case SpvOpReadPipe:
   case SpvOpWritePipe: {
      const bool write = opcode == SpvOpWritePipe;
      nir_def *pipe = vtn_get_nir_ssa(b, w[3]);
      nir_def *r = vtn_pipe_reserve(b, pipe, nir_imm_int(nb, 1), SCOPE_INVOCATION, write);
      nir_def *ok = nir_ine_imm(nb, nir_unpack_64_2x32_split_y(nb, r), 0);
      vtn_pipe_copy(b, pipe, nir_unpack_64_2x32_split_x(nb, r), ok, w[4],
                    vtn_constant_uint(b, w[5]), vtn_constant_uint(b, w[6]), write);
      res = nir_bcsel(nb, ok, nir_imm_int(nb, 0), nir_imm_int(nb, -1));
      break;
   }

   case SpvOpReservedReadPipe:
   case SpvOpReservedWritePipe: {
      const bool write = opcode == SpvOpReservedWritePipe;
      nir_def *pipe = vtn_get_nir_ssa(b, w[3]);
      nir_def *r = vtn_get_nir_ssa(b, w[4]);
      nir_def *ok = nir_ine_imm(nb, nir_unpack_64_2x32_split_y(nb, r), 0);
      nir_def *index = nir_iadd(nb, nir_unpack_64_2x32_split_x(nb, r),
                                nir_u2u32(nb, vtn_get_nir_ssa(b, w[5])));
      vtn_pipe_copy(b, pipe, index, ok, w[6],
                    vtn_constant_uint(b, w[7]), vtn_constant_uint(b, w[8]), write);
      res = nir_bcsel(nb, ok, nir_imm_int(nb, 0), nir_imm_int(nb, -1));
      break;
   }

   case SpvOpReserveReadPipePackets:
   case SpvOpReserveWritePipePackets:
      res = vtn_pipe_reserve(b, vtn_get_nir_ssa(b, w[3]), vtn_get_nir_ssa(b, w[4]),
                             SCOPE_INVOCATION, opcode == SpvOpReserveWritePipePackets);
      break;

   case SpvOpGroupReserveReadPipePackets:
   case SpvOpGroupReserveWritePipePackets:
      vtn_fail_if(vtn_constant_uint(b, w[3]) != SpvScopeWorkgroup,
                  "only work-group scope pipe reservations are supported");
      res = vtn_pipe_reserve(b, vtn_get_nir_ssa(b, w[4]), vtn_get_nir_ssa(b, w[5]),
                             SCOPE_WORKGROUP, opcode == SpvOpGroupReserveWritePipePackets);
      break;

   case SpvOpCommitReadPipe:
   case SpvOpCommitWritePipe:
   case SpvOpGroupCommitReadPipe:
   case SpvOpGroupCommitWritePipe:
      /* packets are visible to the next kernel anyway */
      return;

   case SpvOpIsValidReserveId:
      res = nir_ine_imm(nb, nir_unpack_64_2x32_split_y(nb, vtn_get_nir_ssa(b, w[3])), 0);
      break;

   case SpvOpGetNumPipePackets: {
      nir_def *pipe = vtn_get_nir_ssa(b, w[3]);
      res = nir_isub(nb, vtn_pipe_load_u32(nb, pipe, 0), vtn_pipe_load_u32(nb, pipe, 4));
      break;
   }

   case SpvOpGetMaxPipePackets:
      res = vtn_pipe_load_u32(nb, vtn_get_nir_ssa(b, w[3]), 8);
      break;

   default:
      UNREACHABLE("not a pipe instruction");
   }

   vtn_push_nir_ssa(b, w[2], res);
}

/* OpenCL 2.0 device-side enqueue. An on-device queue is the global address
 * of a buffer (the frontend keeps the same layout, rusticl core/queue.rs):
 *
 *   header (256 bytes): uint32 used, capacity, next_event, valid
 *   uint32 event_status[1024]   user event status, ~0 while not set
 *   spare record (4096 bytes)   target of records that do not fit
 *   records
 *
 * A kernel appends a record per enqueue_kernel / enqueue_marker /
 * capture_event_profiling_info: an atomic add on used reserves it, no
 * control flow is needed (a record that does not fit goes to the spare
 * one). The host launches the recorded kernels once the kernel is done,
 * as the Arm blob does. The block literal is copied into the record; the
 * block's kernel gets its address as its first argument.
 *
 * Record: uint32 size, kind (1 kernel, 2 marker, 3 profiling), invoke (hash
 * of the block kernel's entry point name), flags, work_dim, event,
 * num_events, num_locals; uint64 offset[3], global[3], local[3];
 * uint32 param_size, pad; uint64 aux (profiling destination), pad;
 * uint32 wait_events[num_events], local_sizes[num_locals]; then aligned to
 * 16 bytes the block literal.
 *
 * Header: uint32 used, capacity, next_event, valid; at EVENTS the status of
 * the user events; at TIMES per event the uint64 execution and completion
 * times in ns the host writes once its kernel is done (0 before).
 */
#define VTN_DEVQ_USED          0
#define VTN_DEVQ_CAPACITY      4
#define VTN_DEVQ_NEXT_EVENT    8
#define VTN_DEVQ_VALID         12
#define VTN_DEVQ_EVENTS        256
#define VTN_DEVQ_MAX_EVENTS    1024
#define VTN_DEVQ_SPARE         (VTN_DEVQ_EVENTS + 4 * VTN_DEVQ_MAX_EVENTS)
#define VTN_DEVQ_SPARE_SIZE    4096
#define VTN_DEVQ_TIMES         (VTN_DEVQ_SPARE + VTN_DEVQ_SPARE_SIZE)
#define VTN_DEVQ_RECORDS       (VTN_DEVQ_TIMES + 16 * VTN_DEVQ_MAX_EVENTS)

#define VTN_DEVQ_REC_SIZE       0
#define VTN_DEVQ_REC_KIND       4
#define VTN_DEVQ_REC_INVOKE     8
#define VTN_DEVQ_REC_FLAGS      12
#define VTN_DEVQ_REC_DIM        16
#define VTN_DEVQ_REC_EVENT      20
#define VTN_DEVQ_REC_NUM_EVENTS 24
#define VTN_DEVQ_REC_NUM_LOCALS 28
#define VTN_DEVQ_REC_OFFSET     32
#define VTN_DEVQ_REC_GLOBAL     56
#define VTN_DEVQ_REC_LOCAL      80
#define VTN_DEVQ_REC_PARAM_SIZE 104
#define VTN_DEVQ_REC_AUX        112
#define VTN_DEVQ_REC_WAIT       128

#define VTN_CLK_ENQUEUE_FAILURE -101
#define VTN_CLK_INVALID_QUEUE   -102

static nir_deref_instr *
vtn_devq_deref(nir_builder *b, nir_def *addr, const struct glsl_type *type,
               unsigned align)
{
   return nir_build_deref_cast_with_alignment(b, addr, nir_var_mem_global,
                                              type, 0, align, 0);
}

static void
vtn_devq_store(nir_builder *b, nir_def *base, unsigned offset, nir_def *val)
{
   nir_def *addr = nir_iadd_imm(b, base, offset);
   nir_store_deref(b, vtn_devq_deref(b, addr, glsl_uintN_t_type(val->bit_size),
                                     val->bit_size / 8),
                   val, 1);
}

static nir_def *
vtn_devq_load_u32(nir_builder *b, nir_def *base, unsigned offset)
{
   nir_deref_instr *d =
      vtn_devq_deref(b, nir_iadd_imm(b, base, offset), glsl_uint_type(), 4);
   return nir_load_deref_with_access(b, d, ACCESS_COHERENT | ACCESS_VOLATILE);
}

static nir_def *
vtn_devq_atomic_add(nir_builder *b, nir_def *base, unsigned offset,
                    nir_def *val)
{
   nir_deref_instr *d =
      vtn_devq_deref(b, nir_iadd_imm(b, base, offset), glsl_uint_type(), 4);
   return nir_deref_atomic(b, 32, &d->def, val,
                           .atomic_op = nir_atomic_op_iadd);
}

/* the queue value of the default queue: 0 without one */
static nir_def *
vtn_devq_default(nir_builder *b)
{
   nir_def *q = nir_load_default_queue_cl(b);
   nir_def *valid = vtn_devq_load_u32(b, q, VTN_DEVQ_VALID);
   return nir_bcsel(b, nir_ine_imm(b, valid, 0), q, nir_imm_int64(b, 0));
}

/* FNV-1a of the entry point name of a block kernel, the host maps it back */
static uint32_t
vtn_devq_invoke_hash(struct vtn_builder *b, uint32_t func_id)
{
   const struct vtn_value *v = vtn_untyped_value(b, func_id);
   const char *name = v->entry_point_name;
   vtn_fail_if(!name, "Invoke %u of an enqueue must be a kernel", func_id);
   uint32_t h = 2166136261u;
   for (const char *c = name; *c; c++)
      h = (h ^ (uint8_t)*c) * 16777619u;
   return h;
}

static bool
vtn_devq_is_null_ptr(struct vtn_builder *b, uint32_t id)
{
   struct vtn_value *v = vtn_untyped_value(b, id);
   return v->value_type == vtn_value_type_constant && v->is_null_constant;
}

/* Reserves a record of size bytes on queue q (0 = invalid). Returns the
 * record address and sets *res to the enqueue result.
 */
static nir_def *
vtn_devq_reserve(nir_builder *b, nir_def *q, unsigned size, nir_def **res)
{
   nir_def *is_null = nir_ieq_imm(b, q, 0);
   nir_def *qe = nir_bcsel(b, is_null, nir_load_default_queue_cl(b), q);
   nir_def *bad = nir_ior(b, is_null,
                          nir_ieq_imm(b, vtn_devq_load_u32(b, qe, VTN_DEVQ_VALID), 0));
   nir_def *off = vtn_devq_atomic_add(b, qe, VTN_DEVQ_USED, nir_imm_int(b, size));
   nir_def *cap = vtn_devq_load_u32(b, qe, VTN_DEVQ_CAPACITY);
   nir_def *fits = nir_uge(b, nir_u2u64(b, cap),
                           nir_iadd_imm(b, nir_u2u64(b, off), size));
   nir_def *ok = nir_iand(b, nir_inot(b, bad), fits);
   *res = nir_bcsel(b, ok, nir_imm_int(b, 0),
                    nir_bcsel(b, bad, nir_imm_int(b, VTN_CLK_INVALID_QUEUE),
                              nir_imm_int(b, VTN_CLK_ENQUEUE_FAILURE)));
   nir_def *rec_off = nir_bcsel(b, ok,
                                nir_iadd_imm(b, nir_u2u64(b, off), VTN_DEVQ_RECORDS),
                                nir_imm_int64(b, VTN_DEVQ_SPARE));
   return nir_iadd(b, qe, rec_off);
}

/* Event ids of the wait list into the record, an event id for ret_event */
static void
vtn_devq_events(struct vtn_builder *b, nir_def *q, nir_def *rec,
                nir_def *ok, uint32_t num_id, uint32_t wait_id, uint32_t ret_id,
                unsigned *num_events)
{
   nir_builder *nb = &b->nb;
   *num_events = vtn_constant_uint(b, num_id);
   vtn_fail_if(*num_events > 64, "Too many events in an enqueue");
   vtn_devq_store(nb, rec, VTN_DEVQ_REC_NUM_EVENTS, nir_imm_int(nb, *num_events));

   if (*num_events) {
      struct vtn_pointer *ptr = vtn_value(b, wait_id, vtn_value_type_pointer)->pointer;
      nir_deref_instr *d = vtn_pointer_to_deref(b, ptr);
      nir_deref_instr *arr =
         nir_build_deref_cast_with_alignment(nb, &d->def, d->modes,
                                             glsl_uint64_t_type(), 8, 8, 0);
      for (unsigned i = 0; i < *num_events; i++) {
         nir_def *e = nir_load_deref(nb, nir_build_deref_ptr_as_array(nb, arr, nir_imm_int64(nb, i)));
         vtn_devq_store(nb, rec, VTN_DEVQ_REC_WAIT + 4 * i, nir_u2u32(nb, e));
      }
   }

   nir_def *event = nir_imm_int(nb, 0);
   if (!vtn_devq_is_null_ptr(b, ret_id)) {
      nir_def *qe = nir_bcsel(nb, nir_ieq_imm(nb, q, 0), nir_load_default_queue_cl(nb), q);
      nir_def *id = vtn_devq_atomic_add(nb, qe, VTN_DEVQ_NEXT_EVENT, nir_imm_int(nb, 1));
      event = nir_bcsel(nb, ok, id, nir_imm_int(nb, 0));
      struct vtn_pointer *ptr = vtn_value(b, ret_id, vtn_value_type_pointer)->pointer;
      nir_deref_instr *d = vtn_pointer_to_deref(b, ptr);
      nir_deref_instr *e =
         nir_build_deref_cast_with_alignment(nb, &d->def, d->modes,
                                             glsl_uint64_t_type(), 8, 8, 0);
      nir_store_deref(nb, e, nir_u2u64(nb, event), 1);
   }
   vtn_devq_store(nb, rec, VTN_DEVQ_REC_EVENT, event);
}

static void
vtn_devq_copy_ndrange(struct vtn_builder *b, nir_def *rec, uint32_t ptr_id)
{
   nir_builder *nb = &b->nb;
   struct vtn_pointer *ptr = vtn_value(b, ptr_id, vtn_value_type_pointer)->pointer;
   nir_deref_instr *nd = vtn_pointer_to_deref(b, ptr);
   vtn_fail_if(!glsl_type_is_struct(nd->type) || glsl_get_length(nd->type) != 4,
               "NDRange must point to an ndrange_t");

   nir_def *dim = nir_load_deref(nb, nir_build_deref_struct(nb, nd, 0));
   vtn_devq_store(nb, rec, VTN_DEVQ_REC_DIM, nir_u2u32(nb, dim));
   static const unsigned offsets[3] = {
      VTN_DEVQ_REC_OFFSET, VTN_DEVQ_REC_GLOBAL, VTN_DEVQ_REC_LOCAL,
   };
   for (unsigned m = 0; m < 3; m++) {
      nir_deref_instr *arr = nir_build_deref_struct(nb, nd, m + 1);
      for (unsigned i = 0; i < 3; i++) {
         nir_def *v = nir_load_deref(nb, nir_build_deref_array_imm(nb, arr, i));
         vtn_devq_store(nb, rec, offsets[m] + 8 * i, nir_u2u64(nb, v));
      }
   }
}

static void
vtn_devq_copy_param(struct vtn_builder *b, nir_def *dst, uint32_t ptr_id,
                    unsigned size, unsigned align)
{
   nir_builder *nb = &b->nb;
   unsigned chunk = 8;
   while (chunk > 1 && (size % chunk || align % chunk))
      chunk /= 2;
   const struct glsl_type *type = glsl_uintN_t_type(chunk * 8);

   struct vtn_pointer *ptr = vtn_value(b, ptr_id, vtn_value_type_pointer)->pointer;
   nir_deref_instr *src = vtn_pointer_to_deref(b, ptr);
   nir_deref_instr *src_cast =
      nir_build_deref_cast_with_alignment(nb, &src->def, src->modes, type,
                                          chunk, chunk, 0);
   nir_deref_instr *dst_cast =
      nir_build_deref_cast_with_alignment(nb, dst, nir_var_mem_global, type,
                                          chunk, chunk, 0);
   for (unsigned i = 0; i < size / chunk; i++) {
      nir_def *v = nir_load_deref(nb, nir_build_deref_ptr_as_array(nb, src_cast, nir_imm_int64(nb, i)));
      nir_store_deref(nb, nir_build_deref_ptr_as_array(nb, dst_cast, nir_imm_int64(nb, i)), v, 1);
   }
}

void
vtn_handle_device_enqueue(struct vtn_builder *b, SpvOp opcode,
                          const uint32_t *w, unsigned count)
{
   nir_builder *nb = &b->nb;

   switch (opcode) {
   case SpvOpGetDefaultQueue:
      vtn_push_nir_ssa(b, w[2], vtn_devq_default(nb));
      return;

   case SpvOpBuildNDRange: {
      struct vtn_type *type = vtn_get_type(b, w[1]);
      vtn_fail_if(!glsl_type_is_struct(type->type) || glsl_get_length(type->type) != 4,
                  "OpBuildNDRange must return an ndrange_t");
      struct vtn_ssa_value *res = vtn_create_ssa_value(b, type->type);
      /* members: work_dim, offset, global size, local size */
      struct vtn_ssa_value *srcs[3] = {
         vtn_ssa_value(b, w[5]), vtn_ssa_value(b, w[3]), vtn_ssa_value(b, w[4]),
      };
      unsigned dim = 1;
      for (unsigned m = 0; m < 3; m++) {
         struct vtn_ssa_value *arr = res->elems[m + 1];
         unsigned bits = glsl_get_bit_size(glsl_get_array_element(arr->type));
         struct vtn_ssa_value *src = srcs[m];
         unsigned n = 1;
         if (glsl_type_is_array(src->type)) {
            n = glsl_get_length(src->type);
            dim = n;
         }
         for (unsigned i = 0; i < 3; i++) {
            nir_def *v;
            if (i >= n)
               v = nir_imm_intN_t(nb, 0, bits);
            else if (glsl_type_is_array(src->type))
               v = nir_u2uN(nb, src->elems[i]->def, bits);
            else
               v = nir_u2uN(nb, src->def, bits);
            arr->elems[i]->def = v;
         }
      }
      res->elems[0]->def = nir_imm_intN_t(nb, dim, glsl_get_bit_size(res->elems[0]->type));
      vtn_push_ssa_value(b, w[2], res);
      return;
   }

   case SpvOpEnqueueKernel: {
      /* Queue, Flags, ND Range, Num Events, Wait Events, Ret Event, Invoke,
       * Param, Param Size, Param Align, Local Size...
       */
      unsigned num_events = vtn_constant_uint(b, w[6]);
      unsigned num_locals = count - 13;
      unsigned param_size = vtn_constant_uint(b, w[11]);
      unsigned param_align = vtn_constant_uint(b, w[12]);
      unsigned lit_off = align(VTN_DEVQ_REC_WAIT + 4 * (num_events + num_locals), 16);
      unsigned size = align(lit_off + param_size, 16);
      vtn_fail_if(size > VTN_DEVQ_SPARE_SIZE, "Block literal too large to enqueue");

      nir_def *q = vtn_get_nir_ssa(b, w[3]);
      nir_def *res;
      nir_def *rec = vtn_devq_reserve(nb, q, size, &res);
      nir_def *ok = nir_ieq_imm(nb, res, 0);
      vtn_devq_store(nb, rec, VTN_DEVQ_REC_SIZE, nir_imm_int(nb, size));
      vtn_devq_store(nb, rec, VTN_DEVQ_REC_KIND, nir_imm_int(nb, 1));
      vtn_devq_store(nb, rec, VTN_DEVQ_REC_INVOKE,
                     nir_imm_int(nb, vtn_devq_invoke_hash(b, w[9])));
      vtn_devq_store(nb, rec, VTN_DEVQ_REC_FLAGS, nir_u2u32(nb, vtn_get_nir_ssa(b, w[4])));
      vtn_devq_store(nb, rec, VTN_DEVQ_REC_NUM_LOCALS, nir_imm_int(nb, num_locals));
      vtn_devq_store(nb, rec, VTN_DEVQ_REC_PARAM_SIZE, nir_imm_int(nb, param_size));
      vtn_devq_copy_ndrange(b, rec, w[5]);
      vtn_devq_events(b, q, rec, ok, w[6], w[7], w[8], &num_events);
      for (unsigned i = 0; i < num_locals; i++) {
         /* The SPIR-V LLVM translator passes a pointer to each local size
          * (an element of the array clang builds for the sizes).
          */
         nir_def *local;
         if (vtn_untyped_value(b, w[13 + i])->value_type == vtn_value_type_pointer) {
            struct vtn_pointer *ptr =
               vtn_value(b, w[13 + i], vtn_value_type_pointer)->pointer;
            local = nir_load_deref(nb, vtn_pointer_to_deref(b, ptr));
         } else {
            local = vtn_get_nir_ssa(b, w[13 + i]);
         }
         vtn_devq_store(nb, rec, VTN_DEVQ_REC_WAIT + 4 * (num_events + i),
                        nir_u2u32(nb, local));
      }
      vtn_devq_copy_param(b, nir_iadd_imm(nb, rec, lit_off), w[10], param_size,
                          param_align);
      vtn_push_nir_ssa(b, w[2], res);
      return;
   }

   case SpvOpEnqueueMarker: {
      /* Queue, Num Events, Wait Events, Ret Event */
      unsigned num_events = vtn_constant_uint(b, w[4]);
      unsigned size = align(VTN_DEVQ_REC_WAIT + 4 * num_events, 16);
      nir_def *q = vtn_get_nir_ssa(b, w[3]);
      nir_def *res;
      nir_def *rec = vtn_devq_reserve(nb, q, size, &res);
      nir_def *ok = nir_ieq_imm(nb, res, 0);
      vtn_devq_store(nb, rec, VTN_DEVQ_REC_SIZE, nir_imm_int(nb, size));
      vtn_devq_store(nb, rec, VTN_DEVQ_REC_KIND, nir_imm_int(nb, 2));
      vtn_devq_store(nb, rec, VTN_DEVQ_REC_NUM_LOCALS, nir_imm_int(nb, 0));
      vtn_devq_events(b, q, rec, ok, w[4], w[5], w[6], &num_events);
      vtn_push_nir_ssa(b, w[2], res);
      return;
   }

   case SpvOpCaptureEventProfilingInfo: {
      /* Event, Profiling Info, Value: the times of a command that is done
       * are written right away; the record makes the host write them once
       * the command is done.
       */
      nir_def *evt = nir_u2u32(nb, vtn_get_nir_ssa(b, w[1]));
      nir_def *slot = nir_imul_imm(nb, nir_iand_imm(nb, evt, VTN_DEVQ_MAX_EVENTS - 1), 16);
      nir_def *times = nir_iadd(nb, nir_load_default_queue_cl(nb), nir_u2u64(nb, slot));
      nir_def *t[2];
      for (unsigned i = 0; i < 2; i++) {
         nir_deref_instr *d = vtn_devq_deref(
            nb, nir_iadd_imm(nb, times, VTN_DEVQ_TIMES + 8 * i), glsl_uint64_t_type(), 8);
         t[i] = nir_load_deref_with_access(nb, d, ACCESS_COHERENT | ACCESS_VOLATILE);
      }
      struct vtn_pointer *vptr = vtn_value(b, w[3], vtn_value_type_pointer)->pointer;
      nir_def *vdst = nir_u2u64(nb, vtn_pointer_to_ssa(b, vptr));
      /* without control flow: before the times exist they go to the spare record */
      vdst = nir_bcsel(nb, nir_ine_imm(nb, t[0], 0), vdst,
                       nir_iadd_imm(nb, nir_load_default_queue_cl(nb), VTN_DEVQ_SPARE));
      vtn_devq_store(nb, vdst, 0, t[0]);
      vtn_devq_store(nb, vdst, 8, t[1]);

      unsigned size = VTN_DEVQ_REC_WAIT;
      nir_def *res;
      nir_def *rec = vtn_devq_reserve(nb, vtn_devq_default(nb), size, &res);
      vtn_devq_store(nb, rec, VTN_DEVQ_REC_SIZE, nir_imm_int(nb, size));
      vtn_devq_store(nb, rec, VTN_DEVQ_REC_KIND, nir_imm_int(nb, 3));
      vtn_devq_store(nb, rec, VTN_DEVQ_REC_EVENT, nir_u2u32(nb, vtn_get_nir_ssa(b, w[1])));
      vtn_devq_store(nb, rec, VTN_DEVQ_REC_FLAGS, nir_u2u32(nb, vtn_get_nir_ssa(b, w[2])));
      vtn_devq_store(nb, rec, VTN_DEVQ_REC_NUM_EVENTS, nir_imm_int(nb, 0));
      vtn_devq_store(nb, rec, VTN_DEVQ_REC_NUM_LOCALS, nir_imm_int(nb, 0));
      struct vtn_pointer *ptr = vtn_value(b, w[3], vtn_value_type_pointer)->pointer;
      nir_def *dst = vtn_pointer_to_ssa(b, ptr);
      vtn_devq_store(nb, rec, VTN_DEVQ_REC_AUX, nir_u2u64(nb, dst));
      return;
   }

   case SpvOpRetainEvent:
   case SpvOpReleaseEvent:
      /* events live as long as the queue's records */
      return;

   case SpvOpCreateUserEvent: {
      nir_def *q = nir_load_default_queue_cl(nb);
      nir_def *id = vtn_devq_atomic_add(nb, q, VTN_DEVQ_NEXT_EVENT, nir_imm_int(nb, 1));
      vtn_push_nir_ssa(b, w[2], nir_u2u64(nb, id));
      return;
   }

   case SpvOpIsValidEvent:
      vtn_push_nir_ssa(b, w[2], nir_ine_imm(nb, vtn_get_nir_ssa(b, w[3]), 0));
      return;

   case SpvOpSetUserEventStatus: {
      nir_def *q = nir_load_default_queue_cl(nb);
      nir_def *id = nir_u2u64(nb, nir_iand_imm(nb, nir_u2u32(nb, vtn_get_nir_ssa(b, w[1])),
                                              VTN_DEVQ_MAX_EVENTS - 1));
      nir_def *addr = nir_iadd(nb, nir_iadd_imm(nb, q, VTN_DEVQ_EVENTS), nir_imul_imm(nb, id, 4));
      vtn_devq_store(nb, addr, 0, nir_u2u32(nb, vtn_get_nir_ssa(b, w[2])));
      return;
   }

   case SpvOpGetKernelWorkGroupSize:
   case SpvOpGetKernelPreferredWorkGroupSizeMultiple: {
      /* Invoke, Param, Param Size, Param Align: a block kernel runs with
       * at most 64 work-items per group (any local size up to it works on
       * every device of the frontend)
       */
      const struct glsl_type *type = vtn_get_type(b, w[1])->type;
      unsigned v = opcode == SpvOpGetKernelWorkGroupSize ? 64 : 1;
      vtn_push_nir_ssa(b, w[2], nir_imm_intN_t(nb, v, glsl_get_bit_size(type)));
      return;
   }

   default:
      vtn_fail_with_opcode("Unhandled opcode", opcode);
   }
}
