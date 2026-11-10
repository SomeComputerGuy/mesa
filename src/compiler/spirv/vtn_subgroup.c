/*
 * Copyright © 2016 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "vtn_private.h"

static struct vtn_ssa_value *
vtn_build_subgroup_instr(struct vtn_builder *b,
                         nir_intrinsic_op nir_op,
                         struct vtn_ssa_value *src0,
                         struct vtn_ssa_value *src1,
                         nir_def *index,
                         unsigned const_idx0,
                         unsigned const_idx1)
{
   /* Some of the subgroup operations take an index.  SPIR-V allows this to be
    * any integer type.  To make things simpler for drivers, we only support
    * 32-bit indices.
    */
   if (index && index->bit_size != 32)
      index = nir_u2u32(&b->nb, index);

   struct vtn_ssa_value *dst = vtn_create_ssa_value(b, src0->type);

   vtn_assert(dst->type == src0->type);
   vtn_assert(glsl_type_is_vector_or_scalar(dst->type));
   if (src1)
      vtn_assert(src1->type == src0->type);

   nir_intrinsic_instr *intrin =
      nir_intrinsic_instr_create(b->nb.shader, nir_op);
   nir_def_init_for_type(&intrin->instr, &intrin->def, dst->type);
   intrin->num_components = intrin->def.num_components;

   intrin->src[0] = nir_src_for_ssa(src0->def);
   if (src1)
      intrin->src[1] = nir_src_for_ssa(src1->def);
   if (index)
      intrin->src[src1 ? 2 : 1] = nir_src_for_ssa(index);

   intrin->const_index[0] = const_idx0;
   intrin->const_index[1] = const_idx1;

   nir_builder_instr_insert(&b->nb, &intrin->instr);

   dst->def = &intrin->def;

   return dst;
}

/* OpenCL work-group collectives (work-group scope OpGroup*): emitted as the
 * workgroup_* intrinsics, which the OpenCL frontend lowers in the shader once
 * the control flow is structured (rusticl_lower_wg_collectives).
 */
static bool
vtn_is_wg_collective(struct vtn_builder *b, SpvOp opcode, const uint32_t *w)
{
   if (!b->options->cl_max_workgroup_invocations)
      return false;

   switch (opcode) {
   case SpvOpGroupAll:
   case SpvOpGroupAny:
   case SpvOpGroupBroadcast:
   case SpvOpGroupIAdd:
   case SpvOpGroupFAdd:
   case SpvOpGroupFMin:
   case SpvOpGroupUMin:
   case SpvOpGroupSMin:
   case SpvOpGroupFMax:
   case SpvOpGroupUMax:
   case SpvOpGroupSMax:
   case SpvOpGroupIMulKHR:
   case SpvOpGroupFMulKHR:
   case SpvOpGroupBitwiseAndKHR:
   case SpvOpGroupBitwiseOrKHR:
   case SpvOpGroupBitwiseXorKHR:
   case SpvOpGroupLogicalAndKHR:
   case SpvOpGroupLogicalOrKHR:
   case SpvOpGroupLogicalXorKHR:
      return vtn_constant_uint(b, w[3]) == SpvScopeWorkgroup;
   default:
      return false;
   }
}

static nir_def *
vtn_wg_reduce_scan(struct vtn_builder *b, SpvGroupOperation gop, nir_op op,
                   nir_def *x)
{
   nir_builder *nb = &b->nb;
   vtn_fail_if(x->num_components != 1, "work-group collective of a vector");
   const bool is_bool = x->bit_size == 1;
   if (is_bool)
      x = nir_b2i32(nb, x);

   nir_def *r;
   switch (gop) {
   case SpvGroupOperationReduce:
      r = nir_workgroup_reduce(nb, x, .reduction_op = op);
      break;
   case SpvGroupOperationInclusiveScan:
      r = nir_workgroup_inclusive_scan(nb, x, .reduction_op = op);
      break;
   case SpvGroupOperationExclusiveScan:
      r = nir_workgroup_exclusive_scan(nb, x, .reduction_op = op);
      break;
   default:
      vtn_fail("work-group collective with group operation %u", gop);
   }
   return is_bool ? nir_ine_imm(nb, r, 0) : r;
}

static void
vtn_handle_wg_collective(struct vtn_builder *b, SpvOp opcode,
                         const uint32_t *w, unsigned count)
{
   nir_op op;

   switch (opcode) {
   case SpvOpGroupAll:
   case SpvOpGroupAny:
      vtn_push_nir_ssa(b, w[2],
         vtn_wg_reduce_scan(b, SpvGroupOperationReduce,
                            opcode == SpvOpGroupAll ? nir_op_iand : nir_op_ior,
                            vtn_get_nir_ssa(b, w[4])));
      return;
   case SpvOpGroupBroadcast: {
      nir_def *x = vtn_get_nir_ssa(b, w[4]);
      vtn_fail_if(x->num_components != 1, "work_group_broadcast of a vector");
      nir_def *id = nir_u2u32(&b->nb, vtn_get_nir_ssa(b, w[5]));
      /* the local id as a vec3, missing components 0 */
      nir_def *c[3];
      for (unsigned i = 0; i < 3; i++)
         c[i] = i < id->num_components ? nir_channel(&b->nb, id, i) : nir_imm_int(&b->nb, 0);
      vtn_push_nir_ssa(b, w[2], nir_workgroup_broadcast(&b->nb, x, nir_vec(&b->nb, c, 3)));
      return;
   }
   case SpvOpGroupIAdd:          op = nir_op_iadd; break;
   case SpvOpGroupFAdd:          op = nir_op_fadd; break;
   case SpvOpGroupFMin:          op = nir_op_fmin; break;
   case SpvOpGroupUMin:          op = nir_op_umin; break;
   case SpvOpGroupSMin:          op = nir_op_imin; break;
   case SpvOpGroupFMax:          op = nir_op_fmax; break;
   case SpvOpGroupUMax:          op = nir_op_umax; break;
   case SpvOpGroupSMax:          op = nir_op_imax; break;
   case SpvOpGroupIMulKHR:       op = nir_op_imul; break;
   case SpvOpGroupFMulKHR:       op = nir_op_fmul; break;
   case SpvOpGroupBitwiseAndKHR: op = nir_op_iand; break;
   case SpvOpGroupBitwiseOrKHR:  op = nir_op_ior;  break;
   case SpvOpGroupBitwiseXorKHR: op = nir_op_ixor; break;
   case SpvOpGroupLogicalAndKHR: op = nir_op_iand; break;
   case SpvOpGroupLogicalOrKHR:  op = nir_op_ior;  break;
   case SpvOpGroupLogicalXorKHR: op = nir_op_ixor; break;
   default:
      UNREACHABLE("not a work-group collective");
   }

   vtn_push_nir_ssa(b, w[2], vtn_wg_reduce_scan(b, (SpvGroupOperation)w[4], op,
                                                vtn_get_nir_ssa(b, w[5])));
}

void
vtn_handle_subgroup(struct vtn_builder *b, SpvOp opcode,
                    const uint32_t *w, unsigned count)
{
   struct vtn_type *dest_type = vtn_get_type(b, w[1]);

   if (vtn_is_wg_collective(b, opcode, w)) {
      vtn_handle_wg_collective(b, opcode, w, count);
      return;
   }

   switch (opcode) {
   case SpvOpGroupNonUniformElect: {
      vtn_fail_if(dest_type->type != glsl_bool_type(),
                  "OpGroupNonUniformElect must return a Bool");
      nir_intrinsic_instr *elect =
         nir_intrinsic_instr_create(b->nb.shader, nir_intrinsic_elect);
      nir_def_init_for_type(&elect->instr, &elect->def, dest_type->type);
      nir_builder_instr_insert(&b->nb, &elect->instr);
      vtn_push_nir_ssa(b, w[2], &elect->def);
      break;
   }

   case SpvOpGroupNonUniformBallot:
   case SpvOpSubgroupBallotKHR: {
      bool has_scope = (opcode != SpvOpSubgroupBallotKHR);
      vtn_fail_if(dest_type->type != glsl_vector_type(GLSL_TYPE_UINT, 4),
                  "OpGroupNonUniformBallot must return a uvec4");
      nir_intrinsic_instr *ballot =
         nir_intrinsic_instr_create(b->nb.shader, nir_intrinsic_ballot);
      ballot->src[0] = nir_src_for_ssa(vtn_get_nir_ssa(b, w[3 + has_scope]));
      nir_def_init(&ballot->instr, &ballot->def, 4, 32);
      ballot->num_components = 4;
      nir_builder_instr_insert(&b->nb, &ballot->instr);
      vtn_push_nir_ssa(b, w[2], &ballot->def);
      break;
   }

   case SpvOpGroupNonUniformInverseBallot: {
      nir_def *dest = nir_inverse_ballot(&b->nb, vtn_get_nir_ssa(b, w[4]));
      vtn_push_nir_ssa(b, w[2], dest);
      break;
   }

   case SpvOpGroupNonUniformBallotBitExtract: {
      nir_def *src0 = vtn_get_nir_ssa(b, w[4]);
      nir_def *src1 = vtn_get_nir_ssa(b, w[5]);
      nir_def *dest = nir_ballot_bitfield_extract(&b->nb, src0, src1);
      vtn_push_nir_ssa(b, w[2], dest);
      break;
   }

   case SpvOpGroupNonUniformBallotBitCount:
   case SpvOpGroupNonUniformBallotFindLSB:
   case SpvOpGroupNonUniformBallotFindMSB: {
      nir_def *dest;
      switch (opcode) {
      case SpvOpGroupNonUniformBallotBitCount: {
         nir_def *src = vtn_get_nir_ssa(b, w[5]);
         switch ((SpvGroupOperation)w[4]) {
         case SpvGroupOperationReduce:
            dest = nir_ballot_bit_count_reduce(&b->nb, src);
            break;
         case SpvGroupOperationInclusiveScan:
            dest = nir_ballot_bit_count_inclusive(&b->nb, src);
            break;
         case SpvGroupOperationExclusiveScan:
            dest = nir_ballot_bit_count_exclusive(&b->nb, src);
            break;
         default:
            UNREACHABLE("Invalid group operation");
         }
         break;
      }
      default: {
         nir_def *src = vtn_get_nir_ssa(b, w[4]);
         switch (opcode) {
         case SpvOpGroupNonUniformBallotFindLSB:
            dest = nir_ballot_find_lsb(&b->nb, src);
            break;
         case SpvOpGroupNonUniformBallotFindMSB:
            dest = nir_ballot_find_msb(&b->nb, src);
            break;
         default:
            UNREACHABLE("Unhandled opcode");
         }
         break;
      }
      }

      dest = nir_i2iN(&b->nb, dest, glsl_get_bit_size(dest_type->type));
      vtn_push_nir_ssa(b, w[2], dest);
      break;
   }

   case SpvOpGroupNonUniformBroadcastFirst:
   case SpvOpSubgroupFirstInvocationKHR: {
      bool has_scope = (opcode != SpvOpSubgroupFirstInvocationKHR);
      vtn_push_ssa_value(b, w[2],
         vtn_build_subgroup_instr(b, nir_intrinsic_read_first_invocation,
                                  vtn_ssa_value(b, w[3 + has_scope]),
                                  NULL, NULL, 0, 0));
      break;
   }

   case SpvOpGroupNonUniformBroadcast:
   case SpvOpGroupBroadcast:
   case SpvOpSubgroupReadInvocationKHR: {
      bool has_scope = (opcode != SpvOpSubgroupReadInvocationKHR);
      vtn_push_ssa_value(b, w[2],
         vtn_build_subgroup_instr(b, nir_intrinsic_read_invocation,
                                  vtn_ssa_value(b, w[3 + has_scope]),
                                  NULL,
                                  vtn_get_nir_ssa(b, w[4 + has_scope]), 0, 0));
      break;
   }

   case SpvOpGroupNonUniformAll:
   case SpvOpGroupNonUniformAny:
   case SpvOpGroupNonUniformAllEqual:
   case SpvOpGroupAll:
   case SpvOpGroupAny:
   case SpvOpSubgroupAllKHR:
   case SpvOpSubgroupAnyKHR:
   case SpvOpSubgroupAllEqualKHR: {
      vtn_fail_if(dest_type->type != glsl_bool_type(),
                  "OpGroupNonUniform(All|Any|AllEqual) must return a bool");
      nir_intrinsic_op op;
      switch (opcode) {
      case SpvOpGroupNonUniformAll:
      case SpvOpGroupAll:
      case SpvOpSubgroupAllKHR:
         op = nir_intrinsic_vote_all;
         break;
      case SpvOpGroupNonUniformAny:
      case SpvOpGroupAny:
      case SpvOpSubgroupAnyKHR:
         op = nir_intrinsic_vote_any;
         break;
      case SpvOpSubgroupAllEqualKHR:
         op = nir_intrinsic_vote_ieq;
         break;
      case SpvOpGroupNonUniformAllEqual:
         switch (glsl_get_base_type(vtn_ssa_value(b, w[4])->type)) {
         case GLSL_TYPE_FLOAT:
         case GLSL_TYPE_FLOAT16:
         case GLSL_TYPE_DOUBLE:
            op = nir_intrinsic_vote_feq;
            break;
         case GLSL_TYPE_UINT:
         case GLSL_TYPE_INT:
         case GLSL_TYPE_UINT8:
         case GLSL_TYPE_INT8:
         case GLSL_TYPE_UINT16:
         case GLSL_TYPE_INT16:
         case GLSL_TYPE_UINT64:
         case GLSL_TYPE_INT64:
         case GLSL_TYPE_BOOL:
            op = nir_intrinsic_vote_ieq;
            break;
         default:
            UNREACHABLE("Unhandled type");
         }
         break;
      default:
         UNREACHABLE("Unhandled opcode");
      }

      nir_def *src0;
      if (opcode == SpvOpGroupNonUniformAll || opcode == SpvOpGroupAll ||
          opcode == SpvOpGroupNonUniformAny || opcode == SpvOpGroupAny ||
          opcode == SpvOpGroupNonUniformAllEqual) {
         src0 = vtn_get_nir_ssa(b, w[4]);
      } else {
         src0 = vtn_get_nir_ssa(b, w[3]);
      }
      nir_intrinsic_instr *intrin =
         nir_intrinsic_instr_create(b->nb.shader, op);
      if (nir_intrinsic_infos[op].src_components[0] == 0)
         intrin->num_components = src0->num_components;
      intrin->src[0] = nir_src_for_ssa(src0);
      nir_def_init_for_type(&intrin->instr, &intrin->def,
                            dest_type->type);
      nir_builder_instr_insert(&b->nb, &intrin->instr);

      vtn_push_nir_ssa(b, w[2], &intrin->def);
      break;
   }

   case SpvOpGroupNonUniformShuffle:
   case SpvOpGroupNonUniformShuffleXor:
   case SpvOpGroupNonUniformShuffleUp:
   case SpvOpGroupNonUniformShuffleDown: {
      nir_intrinsic_op op;
      switch (opcode) {
      case SpvOpGroupNonUniformShuffle:
         op = nir_intrinsic_shuffle;
         break;
      case SpvOpGroupNonUniformShuffleXor:
         op = nir_intrinsic_shuffle_xor;
         break;
      case SpvOpGroupNonUniformShuffleUp:
         op = nir_intrinsic_shuffle_up;
         break;
      case SpvOpGroupNonUniformShuffleDown:
         op = nir_intrinsic_shuffle_down;
         break;
      default:
         UNREACHABLE("Invalid opcode");
      }
      vtn_push_ssa_value(b, w[2],
         vtn_build_subgroup_instr(b, op, vtn_ssa_value(b, w[4]),
                                  NULL,
                                  vtn_get_nir_ssa(b, w[5]), 0, 0));
      break;
   }

   case SpvOpSubgroupShuffleINTEL:
   case SpvOpSubgroupShuffleXorINTEL: {
      nir_intrinsic_op op = opcode == SpvOpSubgroupShuffleINTEL ?
         nir_intrinsic_shuffle : nir_intrinsic_shuffle_xor;
      vtn_push_ssa_value(b, w[2],
         vtn_build_subgroup_instr(b, op, vtn_ssa_value(b, w[3]),
                                  NULL,
                                  vtn_get_nir_ssa(b, w[4]), 0, 0));
      break;
   }

   case SpvOpSubgroupShuffleUpINTEL:
   case SpvOpSubgroupShuffleDownINTEL: {
      nir_intrinsic_op op = opcode == SpvOpSubgroupShuffleUpINTEL ?
         nir_intrinsic_shuffle_up_intel : nir_intrinsic_shuffle_down_intel;
      vtn_push_ssa_value(b, w[2],
         vtn_build_subgroup_instr(b, op,
                                  vtn_ssa_value(b, w[3]),
                                  vtn_ssa_value(b, w[4]),
                                  vtn_get_nir_ssa(b, w[5]),
                                  0, 0));
      break;
   }

   case SpvOpGroupNonUniformRotateKHR: {
      const uint32_t cluster_size = count > 6 ? vtn_constant_uint(b, w[6]) : 0;
      vtn_fail_if(cluster_size && !IS_POT(cluster_size),
                  "Behavior is undefined unless ClusterSize is at least 1 and a power of 2.");

      struct vtn_ssa_value *value = vtn_ssa_value(b, w[4]);
      struct vtn_ssa_value *delta = vtn_ssa_value(b, w[5]);
      vtn_push_nir_ssa(b, w[2],
         vtn_build_subgroup_instr(b, nir_intrinsic_rotate,
                                  value, NULL, delta->def, cluster_size, 0)->def);
      break;
   }

   case SpvOpGroupNonUniformQuadBroadcast:
      /* From the Vulkan spec 1.3.269:
       *
       * 9.27. Quad Group Operations:
       * "Fragment shaders that statically execute quad group operations
       * must launch sufficient invocations to ensure their correct operation;"
       */
      if (b->shader->info.stage == MESA_SHADER_FRAGMENT)
         b->shader->info.fs.require_full_quads = true;

      vtn_push_ssa_value(b, w[2],
         vtn_build_subgroup_instr(b, nir_intrinsic_quad_broadcast,
                                  vtn_ssa_value(b, w[4]),
                                  NULL,
                                  vtn_get_nir_ssa(b, w[5]), 0, 0));
      break;

   case SpvOpGroupNonUniformQuadSwap: {
      if (b->shader->info.stage == MESA_SHADER_FRAGMENT)
         b->shader->info.fs.require_full_quads = true;

      unsigned direction = vtn_constant_uint(b, w[5]);
      nir_intrinsic_op op;
      switch (direction) {
      case 0:
         op = nir_intrinsic_quad_swap_horizontal;
         break;
      case 1:
         op = nir_intrinsic_quad_swap_vertical;
         break;
      case 2:
         op = nir_intrinsic_quad_swap_diagonal;
         break;
      default:
         vtn_fail("Invalid constant value in OpGroupNonUniformQuadSwap");
      }
      vtn_push_ssa_value(b, w[2],
         vtn_build_subgroup_instr(b, op, vtn_ssa_value(b, w[4]), NULL, NULL,
                                  0, 0));
      break;
   }

   case SpvOpGroupNonUniformQuadAllKHR: {
      nir_def *dest = nir_quad_vote_all(&b->nb, 1, vtn_get_nir_ssa(b, w[3]));
      vtn_push_nir_ssa(b, w[2], dest);
      break;
   }
   case SpvOpGroupNonUniformQuadAnyKHR: {
      nir_def *dest = nir_quad_vote_any(&b->nb, 1, vtn_get_nir_ssa(b, w[3]));
      vtn_push_nir_ssa(b, w[2], dest);
      break;
   }

   case SpvOpGroupNonUniformIAdd:
   case SpvOpGroupNonUniformFAdd:
   case SpvOpGroupNonUniformIMul:
   case SpvOpGroupNonUniformFMul:
   case SpvOpGroupNonUniformSMin:
   case SpvOpGroupNonUniformUMin:
   case SpvOpGroupNonUniformFMin:
   case SpvOpGroupNonUniformSMax:
   case SpvOpGroupNonUniformUMax:
   case SpvOpGroupNonUniformFMax:
   case SpvOpGroupNonUniformBitwiseAnd:
   case SpvOpGroupNonUniformBitwiseOr:
   case SpvOpGroupNonUniformBitwiseXor:
   case SpvOpGroupNonUniformLogicalAnd:
   case SpvOpGroupNonUniformLogicalOr:
   case SpvOpGroupNonUniformLogicalXor:
   case SpvOpGroupIAdd:
   case SpvOpGroupFAdd:
   case SpvOpGroupFMin:
   case SpvOpGroupUMin:
   case SpvOpGroupSMin:
   case SpvOpGroupFMax:
   case SpvOpGroupUMax:
   case SpvOpGroupSMax:
   case SpvOpGroupIMulKHR:
   case SpvOpGroupFMulKHR:
   case SpvOpGroupBitwiseAndKHR:
   case SpvOpGroupBitwiseOrKHR:
   case SpvOpGroupBitwiseXorKHR:
   case SpvOpGroupLogicalAndKHR:
   case SpvOpGroupLogicalOrKHR:
   case SpvOpGroupLogicalXorKHR:
   case SpvOpGroupIAddNonUniformAMD:
   case SpvOpGroupFAddNonUniformAMD:
   case SpvOpGroupFMinNonUniformAMD:
   case SpvOpGroupUMinNonUniformAMD:
   case SpvOpGroupSMinNonUniformAMD:
   case SpvOpGroupFMaxNonUniformAMD:
   case SpvOpGroupUMaxNonUniformAMD:
   case SpvOpGroupSMaxNonUniformAMD: {
      nir_op reduction_op;
      switch (opcode) {
      case SpvOpGroupNonUniformIAdd:
      case SpvOpGroupIAdd:
      case SpvOpGroupIAddNonUniformAMD:
         reduction_op = nir_op_iadd;
         break;
      case SpvOpGroupNonUniformFAdd:
      case SpvOpGroupFAdd:
      case SpvOpGroupFAddNonUniformAMD:
         reduction_op = nir_op_fadd;
         break;
      case SpvOpGroupIMulKHR:
      case SpvOpGroupNonUniformIMul:
         reduction_op = nir_op_imul;
         break;
      case SpvOpGroupFMulKHR:
      case SpvOpGroupNonUniformFMul:
         reduction_op = nir_op_fmul;
         break;
      case SpvOpGroupNonUniformSMin:
      case SpvOpGroupSMin:
      case SpvOpGroupSMinNonUniformAMD:
         reduction_op = nir_op_imin;
         break;
      case SpvOpGroupNonUniformUMin:
      case SpvOpGroupUMin:
      case SpvOpGroupUMinNonUniformAMD:
         reduction_op = nir_op_umin;
         break;
      case SpvOpGroupNonUniformFMin:
      case SpvOpGroupFMin:
      case SpvOpGroupFMinNonUniformAMD:
         reduction_op = nir_op_fmin;
         break;
      case SpvOpGroupNonUniformSMax:
      case SpvOpGroupSMax:
      case SpvOpGroupSMaxNonUniformAMD:
         reduction_op = nir_op_imax;
         break;
      case SpvOpGroupNonUniformUMax:
      case SpvOpGroupUMax:
      case SpvOpGroupUMaxNonUniformAMD:
         reduction_op = nir_op_umax;
         break;
      case SpvOpGroupNonUniformFMax:
      case SpvOpGroupFMax:
      case SpvOpGroupFMaxNonUniformAMD:
         reduction_op = nir_op_fmax;
         break;
      case SpvOpGroupBitwiseAndKHR:
      case SpvOpGroupLogicalAndKHR:
      case SpvOpGroupNonUniformBitwiseAnd:
      case SpvOpGroupNonUniformLogicalAnd:
         reduction_op = nir_op_iand;
         break;
      case SpvOpGroupBitwiseOrKHR:
      case SpvOpGroupLogicalOrKHR:
      case SpvOpGroupNonUniformBitwiseOr:
      case SpvOpGroupNonUniformLogicalOr:
         reduction_op = nir_op_ior;
         break;
      case SpvOpGroupBitwiseXorKHR:
      case SpvOpGroupLogicalXorKHR:
      case SpvOpGroupNonUniformBitwiseXor:
      case SpvOpGroupNonUniformLogicalXor:
         reduction_op = nir_op_ixor;
         break;
      default:
         UNREACHABLE("Invalid reduction operation");
      }

      nir_intrinsic_op op;
      unsigned cluster_size = 0;
      switch ((SpvGroupOperation)w[4]) {
      case SpvGroupOperationReduce:
         op = nir_intrinsic_reduce;
         break;
      case SpvGroupOperationInclusiveScan:
         op = nir_intrinsic_inclusive_scan;
         break;
      case SpvGroupOperationExclusiveScan:
         op = nir_intrinsic_exclusive_scan;
         break;
      case SpvGroupOperationClusteredReduce:
         op = nir_intrinsic_reduce;
         assert(count == 7);
         cluster_size = vtn_constant_uint(b, w[6]);
         break;
      default:
         UNREACHABLE("Invalid group operation");
      }

      vtn_push_ssa_value(b, w[2],
         vtn_build_subgroup_instr(b, op, vtn_ssa_value(b, w[5]), NULL, NULL,
                                  reduction_op, cluster_size));
      break;
   }

   default:
      UNREACHABLE("Invalid SPIR-V opcode");
   }
}
