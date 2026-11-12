/*
 * Copyright (C) 2025 Arm Ltd.
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates.
 * Copyright (C) 2019-2022 Collabora, Ltd.
 * Copyright (C) 2019 Red Hat Inc.
 * Copyright (C) 2018 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "pan_shader.h"
#include <math.h>
#include "nir/tgsi_to_nir.h"
#include "util/u_memory.h"
#include "util/u_prim.h"
#include "nir_builder.h"
#include "nir_serialize.h"
#include "nir_xfb_info.h"
#include "pan_bo.h"
#include "pan_context.h"
#include "pan_compiler.h"
#include "pan_nir.h"
#include "pan_trace.h"
#include "compiler/bifrost/bifrost_compile.h"
#include "panfrost/compiler/midgard/midgard_quirks.h"
#include "shader_enums.h"

static struct panfrost_uncompiled_shader *
panfrost_alloc_shader(const nir_shader *nir)
{
   struct panfrost_uncompiled_shader *so =
      rzalloc(NULL, struct panfrost_uncompiled_shader);

   simple_mtx_init(&so->lock, mtx_plain);
   util_dynarray_init(&so->variants, so);

   so->nir = nir;

   /* Serialize the NIR to a binary blob that we can hash for the disk
    * cache. Drop unnecessary information (like variable names) so the
    * serialized NIR is smaller, and also to let us detect more isomorphic
    * shaders when hashing, increasing cache hits.
    */
   struct blob blob;
   blob_init(&blob);
   nir_serialize(&blob, nir, true);
   _mesa_blake3_compute(blob.data, blob.size, so->nir_blake3);
   blob_finish(&blob);

   return so;
}

static struct panfrost_compiled_shader *
panfrost_alloc_variant(struct panfrost_uncompiled_shader *so)
{
   return util_dynarray_grow(&so->variants, struct panfrost_compiled_shader, 1);
}

static bool
lower_load_poly_line_smooth_enabled(nir_builder *b, nir_intrinsic_instr *intrin,
                                    void *data)
{
   if (intrin->intrinsic != nir_intrinsic_load_poly_line_smooth_enabled)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);
   nir_def_replace(&intrin->def, nir_imm_true(b));
   return true;
}

/* From the OpenGL 4.6 spec 14.3.1:
 *
 *    If MULTISAMPLE is disabled, multisample rasterization of all primitives
 *    is equivalent to single-sample (fragment-center) rasterization, except
 *    that the fragment coverage value is set to full coverage.
 *
 * So always use the original sample mask when multisample is disabled */
static bool
lower_sample_mask_writes(nir_builder *b, nir_intrinsic_instr *intrin,
                         void *data)
{
   if (intrin->intrinsic != nir_intrinsic_store_output)
      return false;

   if (nir_intrinsic_io_semantics(intrin).location != FRAG_RESULT_SAMPLE_MASK)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);

   nir_def *orig = nir_load_sample_mask(b);
   nir_def *new =
      nir_bcsel_pan(b, nir_load_multisampled_pan(b), intrin->src[0].ssa, orig);
   nir_src_rewrite(&intrin->src[0], new);

   return true;
}

static bool lower_fs_layer_cull(nir_shader *nir);

static void
panfrost_shader_compile(struct panfrost_screen *screen, const nir_shader *ir,
                        struct util_debug_callback *dbg,
                        const struct pan_varying_layout *varying_layout,
                        struct panfrost_shader_key *key, unsigned req_local_mem,
                        struct panfrost_shader_binary *out)
{
   PAN_TRACE_FUNC(PAN_TRACE_GL_SHADER);

   struct panfrost_device *dev = pan_device(&screen->base);

   nir_shader *s = nir_shader_clone(NULL, ir);

   /* While graphics shaders are preprocessed at CSO create time, compute
    * kernels are not preprocessed until they're cloned since the driver does
    * not get ownership of the NIR from compute CSOs. Do this preprocessing now.
    * Compute CSOs call this function during create time, so preprocessing
    * happens at CSO create time regardless.
    */
   if (mesa_shader_stage_is_compute(s->info.stage)) {
      pan_preprocess_nir(s, dev->arch);
   }

   struct pan_compile_inputs inputs = {
      .arch = dev->arch,
      .model = dev->model,
   };

   /* Lower this early so the backends don't have to worry about it */
   if (s->info.stage == MESA_SHADER_VERTEX && key->vs.outputs_to_memory) {
      /* Feeds a geometry/tessellation control shader: runs as a compute job
       * storing its outputs to memory, no varyings.
       */
      NIR_PASS(_, s, panfrost_nir_lower_vs_to_memory);
      inputs.no_idvs = true;
   } else if (s->info.stage == MESA_SHADER_VERTEX) {
      /* No IDVS for internal XFB shaders. Tessellation evaluation shaders
       * run in a vertex job whose size the GPU patches.
       */
      inputs.no_idvs = s->info.has_transform_feedback_varyings ||
                       s->info.vs.tes_poly || key->vs.no_idvs;

      if (s->info.has_transform_feedback_varyings) {
         static const nir_lower_xfb_to_stores_options xfb_options = {
            .address_format = nir_address_format_64bit_global,
         };

         NIR_PASS(_, s, nir_opt_constant_folding);
         NIR_PASS(_, s, nir_io_add_intrinsic_xfb_info);
         NIR_PASS(_, s, nir_lower_xfb_to_stores, &xfb_options);
      }
   }

   inputs.varying_layout = varying_layout;

   if (s->info.stage == MESA_SHADER_FRAGMENT) {
      inputs.fragcolor_nr_cbufs = key->fs.nr_cbufs_for_fragcolor;

      if (key->fs.sprite_coord_enable) {
         NIR_PASS(_, s, nir_lower_texcoord_replace_late,
                  key->fs.sprite_coord_enable,
                  true /* point coord is sysval */);
         /* Lower load_point_coord if present */
         NIR_PASS(_, s, pan_nir_lower_var_special_pan);
      }

      if (key->fs.clip_plane_enable) {
         NIR_PASS(_, s, nir_lower_clip_fs, key->fs.clip_plane_enable,
                  false, true);
      }

      if (key->fs.layer_cull)
         NIR_PASS(_, s, lower_fs_layer_cull);

      if (key->fs.line_smooth) {
         NIR_PASS(_, s, nir_lower_poly_line_smooth, 16);
         NIR_PASS(_, s, nir_shader_intrinsics_pass,
                  lower_load_poly_line_smooth_enabled,
                  nir_metadata_control_flow, key);
         NIR_PASS(_, s, nir_lower_alu);
      }
   }

   if (dev->arch <= 5 && s->info.stage == MESA_SHADER_FRAGMENT) {
      if (key->fs.nr_cbufs_for_fragcolor) {
         NIR_PASS(_, s, panfrost_nir_remove_fragcolor_stores,
                  key->fs.nr_cbufs_for_fragcolor);
      }

      NIR_PASS(_, s, pan_nir_lower_framebuffer, key->fs.rt_formats,
               pan_raw_format_mask_midgard(key->fs.rt_formats), 0,
               midgard_get_quirks(dev->model->gpu_prod_id) & MIDGARD_BROKEN_BLEND_LOADS);
   }

   /* Lower resource indices */
   NIR_PASS(_, s, panfrost_nir_lower_res_indices, dev->arch);

   pan_postprocess_nir(s, &inputs, &out->info);

   if (s->info.stage == MESA_SHADER_VERTEX) {
      NIR_PASS(_, s, nir_inline_sysval,
               nir_intrinsic_load_noperspective_varyings_pan,
               key->vs.noperspective_varyings);
   }

   if (dev->arch >= 9 && mesa_shader_stage_is_compute(s->info.stage)) {
      out->info.cs.allow_merging_workgroups = valhall_can_merge_workgroups(s);
   }

   NIR_PASS(_, s, panfrost_nir_lower_sysvals, dev->arch, &out->sysvals);

   /* For now, we only allow pushing the default UBO 0, and the sysval UBO (if
    * present). Both of these are mapped on the CPU, but other UBOs are not.
    * When we switch to pushing UBOs with a compute kernel (or CSF instructions)
    * we can relax this. */
   assert(s->info.first_ubo_is_default_ubo);
   inputs.fau.pushable_ubos = BITFIELD_BIT(0);

   if (out->sysvals.sysval_count != 0) {
      inputs.fau.pushable_ubos |= BITFIELD_BIT(PAN_UBO_SYSVALS);
   }

   inputs.fau.promote_immediates = true;

   if (dev->arch >= 9) {
      /* Always enable this for GL, it avoids crashes when using unbound
       * resources. */
      inputs.robust_descriptors = true;
   }

   out->binary = UTIL_DYNARRAY_INIT;
   screen->vtbl.compile_shader(s, &inputs, &out->binary, &out->info);

   /* Report stats only if we really got the shader compiled */
   if (out->binary.size > 0) {
      if (s->info.stage == MESA_SHADER_VERTEX &&
          out->info.vs.secondary_offset) {
         pan_stats_util_debug(dbg, "MESA_SHADER_POSITION",
                              &out->info.stats);
         pan_stats_util_debug(dbg, "MESA_SHADER_VERTEX",
                              &out->info.stats_idvs_varying);
      } else {
         pan_stats_util_debug(dbg, mesa_shader_stage_name(s->info.stage),
                              &out->info.stats);
      }
   }

   assert(req_local_mem >= out->info.wls_size);
   out->info.wls_size = req_local_mem;

   /* In both clone and tgsi_to_nir paths, the shader is ralloc'd against
    * a NULL context
    */
   ralloc_free(s);
}

static void
panfrost_shader_get(struct pipe_screen *pscreen,
                    struct panfrost_pool *shader_pool,
                    struct panfrost_pool *desc_pool,
                    struct panfrost_uncompiled_shader *uncompiled,
                    struct util_debug_callback *dbg,
                    struct panfrost_compiled_shader *state,
                    unsigned req_local_mem)
{
   struct panfrost_screen *screen = pan_screen(pscreen);
   struct panfrost_device *dev = pan_device(pscreen);

   struct panfrost_shader_binary res = {0};

   /* Try to retrieve the variant from the disk cache. If that fails,
    * compile a new variant and store in the disk cache for later reuse.
    */
   if (!panfrost_disk_cache_retrieve(screen->disk_cache, uncompiled,
                                     &state->key, &res)) {

      /* Only use the varying_layout for FS if the key agrees */
      bool use_layout = uncompiled->nir->info.stage != MESA_SHADER_FRAGMENT ||
                        state->key.fs.vs_varying_layout.known != 0;
      const struct pan_varying_layout *varying_layout =
         use_layout ? &uncompiled->vs_varying_layout : NULL;
      panfrost_shader_compile(screen, uncompiled->nir, dbg, varying_layout,
                              &state->key, req_local_mem, &res);

      panfrost_disk_cache_store(screen->disk_cache, uncompiled, &state->key,
                                &res);
   }

   state->info = res.info;
   state->sysvals = res.sysvals;

   if (res.binary.size) {
      state->bin = panfrost_pool_take_ref(
         shader_pool,
         pan_pool_upload_aligned(&shader_pool->base, res.binary.data,
                                 res.binary.size, 128));
   }

   util_dynarray_fini(&res.binary);

   /* Don't upload RSD for fragment shaders since they need draw-time
    * merging for e.g. depth/stencil/alpha. RSDs are replaced by simpler
    * shader program descriptors on Valhall, which can be preuploaded even
    * for fragment shaders. */
   bool upload =
      !(uncompiled->nir->info.stage == MESA_SHADER_FRAGMENT && dev->arch <= 7);
   screen->vtbl.prepare_shader(state, desc_pool, upload);

   panfrost_analyze_sysvals(state);
}

static void
panfrost_build_vs_key(struct panfrost_context *ctx,
                      struct panfrost_vs_key *key,
                      struct panfrost_uncompiled_shader *uncompiled)
{
   struct panfrost_uncompiled_shader *fs = ctx->uncompiled[MESA_SHADER_FRAGMENT];

   assert(fs != NULL && "too early");
   key->noperspective_varyings = fs->noperspective_varyings;
}

static void
panfrost_build_fs_key(struct panfrost_context *ctx,
                      struct panfrost_fs_key *key,
                      struct panfrost_uncompiled_shader *uncompiled)
{
   const nir_shader *nir = uncompiled->nir;

   struct panfrost_device *dev = pan_device(ctx->base.screen);
   struct pipe_framebuffer_state *fb = &ctx->pipe_framebuffer;
   struct pipe_rasterizer_state *rast = (void *)ctx->rasterizer;

   /* gl_FragColor lowering needs the number of colour buffers */
   if (uncompiled->fragcolor_lowered) {
      key->nr_cbufs_for_fragcolor = fb->nr_cbufs;
   }

   /* Point sprite lowering needed on Bifrost and newer */
   if (dev->arch >= 6 && rast && ctx->active_prim == MESA_PRIM_POINTS) {
      key->sprite_coord_enable = rast->sprite_coord_enable;
   }

   /* User clip plane lowering needed everywhere */
   if (rast) {
      key->clip_plane_enable = rast->clip_plane_enable;

      if (u_reduced_prim(ctx->active_prim) == MESA_PRIM_LINES)
         key->line_smooth = rast->line_smooth;
   }

   if (!key->clip_plane_enable) {
      key->vs_varying_layout = uncompiled->vs_varying_layout;

      /* Layered draw: cull by the layer varying of the provoking vertex */
      key->layer_cull = ctx->poly.layer != PAN_LAYER_ALL &&
                        pan_varying_layout_find_slot(&key->vs_varying_layout,
                                                     PAN_LAYER_CULL_SLOT);
   }

   if (dev->arch <= 5) {
      u_foreach_bit(i, (nir->info.outputs_read >> FRAG_RESULT_DATA0)) {
         enum pipe_format fmt = PIPE_FORMAT_R8G8B8A8_UNORM;

         if ((fb->nr_cbufs > i) && fb->cbufs[i].texture)
            fmt = fb->cbufs[i].format;

         if (pan_blendable_formats_v6[fmt].internal)
            fmt = PIPE_FORMAT_NONE;

         key->rt_formats[i] = fmt;
      }
   }
}

static void
panfrost_build_key(struct panfrost_context *ctx,
                   struct panfrost_shader_key *key,
                   struct panfrost_uncompiled_shader *uncompiled)
{
   const nir_shader *nir = uncompiled->nir;

   switch (nir->info.stage) {
   case MESA_SHADER_VERTEX:
      panfrost_build_vs_key(ctx, &key->vs, uncompiled);
      break;
   case MESA_SHADER_FRAGMENT:
      panfrost_build_fs_key(ctx, &key->fs, uncompiled);
      break;
   default:
      break;
   }
}

static struct panfrost_compiled_shader *
panfrost_new_variant_locked(struct panfrost_context *ctx,
                            struct panfrost_uncompiled_shader *uncompiled,
                            const struct panfrost_shader_key *key)
{
   struct panfrost_device *dev = pan_device(ctx->base.screen);
   struct panfrost_compiled_shader *prog = panfrost_alloc_variant(uncompiled);

   *prog = (struct panfrost_compiled_shader){
      .key = *key,
      .stream_output = uncompiled->stream_output,
   };

   panfrost_shader_get(ctx->base.screen, &ctx->shaders, &ctx->descs, uncompiled,
                       &ctx->base.debug, prog, 0);

   prog->earlyzs = pan_earlyzs_analyze(&prog->info, dev->arch);

   return prog;
}

struct panfrost_compiled_shader *
panfrost_get_variant(struct panfrost_context *ctx,
                     struct panfrost_uncompiled_shader *uncompiled,
                     const struct panfrost_shader_key *key)
{
   struct panfrost_compiled_shader *compiled = NULL;

   simple_mtx_lock(&uncompiled->lock);

   util_dynarray_foreach(&uncompiled->variants, struct panfrost_compiled_shader,
                         so) {
      if (memcmp(key, &so->key, sizeof(*key)) == 0) {
         compiled = so;
         break;
      }
   }

   if (compiled == NULL)
      compiled = panfrost_new_variant_locked(ctx, uncompiled, key);

   simple_mtx_unlock(&uncompiled->lock);
   return compiled;
}

static void
panfrost_bind_shader_state(struct pipe_context *pctx, void *hwcso,
                           mesa_shader_stage type)
{
   struct panfrost_context *ctx = pan_context(pctx);
   ctx->uncompiled[type] = hwcso;
   ctx->prog[type] = NULL;

   ctx->dirty |= PAN_DIRTY_TLS_SIZE;
   ctx->dirty_shader[type] |= PAN_DIRTY_STAGE_SHADER;

   if (hwcso)
      panfrost_update_shader_variant(ctx, type);
}

void
panfrost_update_shader_variant(struct panfrost_context *ctx,
                               mesa_shader_stage type)
{
   /* No shader variants for compute. Geometry and tessellation shaders are
    * lowered to programs selected at draw time.
    */
   if (type != MESA_SHADER_VERTEX && type != MESA_SHADER_FRAGMENT)
      return;

   /* We need linking information, defer this */
   if ((type == MESA_SHADER_FRAGMENT && !ctx->uncompiled[MESA_SHADER_VERTEX]) ||
       (type == MESA_SHADER_VERTEX && !ctx->uncompiled[MESA_SHADER_FRAGMENT]))
      return;

   /* Also defer, happens with GALLIUM_HUD */
   if (!ctx->uncompiled[type])
      return;

   /* Match the appropriate variant */
   struct panfrost_uncompiled_shader *uncompiled = ctx->uncompiled[type];
   struct panfrost_compiled_shader *compiled = NULL;

   simple_mtx_lock(&uncompiled->lock);

   struct panfrost_shader_key key = {0};
   panfrost_build_key(ctx, &key, uncompiled);

   util_dynarray_foreach(&uncompiled->variants, struct panfrost_compiled_shader,
                         so) {
      if (memcmp(&key, &so->key, sizeof(key)) == 0) {
         compiled = so;
         break;
      }
   }

   if (compiled == NULL)
      compiled = panfrost_new_variant_locked(ctx, uncompiled, &key);

   ctx->prog[type] = compiled;

   simple_mtx_unlock(&uncompiled->lock);
}

static void
panfrost_bind_vs_state(struct pipe_context *pctx, void *hwcso)
{
   panfrost_bind_shader_state(pctx, hwcso, MESA_SHADER_VERTEX);

   /* Fragment shaders are linked with vertex shaders */
   struct panfrost_context *ctx = pan_context(pctx);
   panfrost_update_shader_variant(ctx, MESA_SHADER_FRAGMENT);
}

static void
panfrost_bind_gs_state(struct pipe_context *pctx, void *hwcso)
{
   panfrost_bind_shader_state(pctx, hwcso, MESA_SHADER_GEOMETRY);
}

static void
panfrost_bind_tcs_state(struct pipe_context *pctx, void *hwcso)
{
   panfrost_bind_shader_state(pctx, hwcso, MESA_SHADER_TESS_CTRL);
}

static void
panfrost_bind_tes_state(struct pipe_context *pctx, void *hwcso)
{
   panfrost_bind_shader_state(pctx, hwcso, MESA_SHADER_TESS_EVAL);
}

static void
panfrost_bind_fs_state(struct pipe_context *pctx, void *hwcso)
{
   panfrost_bind_shader_state(pctx, hwcso, MESA_SHADER_FRAGMENT);

   /* Vertex shaders are linked with fragment shaders */
   struct panfrost_context *ctx = pan_context(pctx);
   panfrost_update_shader_variant(ctx, MESA_SHADER_VERTEX);
}

static unsigned
glsl_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

static struct panfrost_shader_key
panfrost_default_shader_key(struct panfrost_uncompiled_shader *so)
{
   struct panfrost_shader_key key = {0};

   if (so->nir->info.stage == MESA_SHADER_FRAGMENT) {
      /* gl_FragColor lowering needs the number of colour buffers on desktop
      * GL, where it acts as an implicit broadcast to all colour buffers.
      *
      * However, gl_FragColor is a legacy feature, so assume that if
      * gl_FragColor is used, there is only a single render target. The
      * implicit broadcast is neither especially useful nor required by GLES.
      */
      if (so->fragcolor_lowered)
         key.fs.nr_cbufs_for_fragcolor = 1;

      key.fs.vs_varying_layout = so->vs_varying_layout;
   }

   return key;
}

static void
panfrost_collect_vs_varying_layout(struct panfrost_device *dev,
                                   struct panfrost_uncompiled_shader *so,
                                   nir_shader *nir)
{
   struct pan_varying_layout *varying_layout = &so->vs_varying_layout;
   pan_varying_collect_formats(varying_layout, nir, dev->arch);
   pan_build_varying_layout_compact(varying_layout, nir, dev->arch);
}

/*
 * Job manager GPUs have no per-vertex layer or viewport index, so the hardware
 * vertex shader of a geometry shader rasterization draw cannot write them.
 * Layered framebuffers are drawn once per layer instead (one batch per layer):
 * vertices whose layer is not the layer being drawn get an infinite position,
 * which culls their primitives, as panvk does.
 */
struct layer_cull_state {
   nir_variable *layer, *pos;
   nir_intrinsic_instr *pos_store;
};

static bool
lower_layer_viewport_store(nir_builder *b, nir_intrinsic_instr *intr,
                           void *data)
{
   struct layer_cull_state *state = data;

   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   gl_varying_slot loc = nir_intrinsic_io_semantics(intr).location;
   b->cursor = nir_before_instr(&intr->instr);

   if (loc == VARYING_SLOT_LAYER) {
      nir_def *v = nir_channel(b, intr->src[0].ssa, nir_intrinsic_component(intr));
      nir_store_var(b, state->layer, nir_u2u32(b, v), 0x1);
   } else if (loc == VARYING_SLOT_POS && state->pos) {
      nir_def *v = intr->src[0].ssa;
      unsigned wrmask = nir_intrinsic_write_mask(intr)
                        << nir_intrinsic_component(intr);
      if (v->num_components < 4) {
         nir_def *old = nir_load_var(b, state->pos);
         nir_def *comps[4];
         for (unsigned i = 0; i < 4; ++i) {
            comps[i] = (wrmask & BITFIELD_BIT(i))
                          ? nir_channel(b, v, i - nir_intrinsic_component(intr))
                          : nir_channel(b, old, i);
         }
         v = nir_vec(b, comps, 4);
      }
      nir_store_var(b, state->pos, v, 0xf);

      /* Re-emitted once at the end of the program */
      if (!state->pos_store) {
         state->pos_store = intr;
         nir_instr_remove(&intr->instr);
         return true;
      }
   } else if (loc != VARYING_SLOT_VIEWPORT) {
      return false;
   }

   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_rast_layer_viewport(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   struct layer_cull_state state = {0};
   bool writes_layer = nir->info.outputs_written & VARYING_BIT_LAYER;

   if (!(nir->info.outputs_written & (VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT)))
      return false;

   nir_builder b = nir_builder_at(nir_before_impl(impl));

   /* The layer goes to the fragment shader in a flat varying when a slot is
    * free, so whole primitives are kept or discarded by the layer of their
    * provoking vertex. Else vertices are culled by their own layer.
    */
   bool layer_varying =
      !(nir->info.outputs_written & BITFIELD64_BIT(PAN_LAYER_CULL_SLOT));

   if (writes_layer) {
      state.layer =
         nir_local_variable_create(impl, glsl_uint_type(), "layer_cull");
      nir_store_var(&b, state.layer, nir_imm_int(&b, 0), 0x1);
   }

   if (writes_layer && !layer_varying &&
       (nir->info.outputs_written & VARYING_BIT_POS)) {
      state.pos =
         nir_local_variable_create(impl, glsl_vec4_type(), "pos_cull");
      nir_store_var(&b, state.pos, nir_imm_vec4(&b, 0, 0, 0, 1), 0xf);
   }

   nir_shader_intrinsics_pass(nir, lower_layer_viewport_store,
                              nir_metadata_none, &state);

   if (state.pos_store) {
      b.cursor = nir_after_impl(impl);

      nir_def *layer = nir_load_var(&b, state.layer);
      nir_def *ref = nir_load_draw_layer_pan(&b);
      nir_def *keep = nir_ior(&b, nir_ieq(&b, layer, ref),
                              nir_ieq_imm(&b, ref, PAN_LAYER_ALL));
      nir_def *inf = nir_imm_vec4(&b, INFINITY, INFINITY, INFINITY, 1.0f);
      nir_def *pos = nir_bcsel(&b, keep, nir_load_var(&b, state.pos), inf);

      if (state.pos_store->src[0].ssa->bit_size == 16)
         pos = nir_f2f16(&b, pos);

      nir_store_output(&b, pos, nir_imm_int(&b, 0),
                       .base = nir_intrinsic_base(state.pos_store),
                       .write_mask = 0xf,
                       .component = 0,
                       .src_type = nir_intrinsic_src_type(state.pos_store),
                       .io_semantics =
                          nir_intrinsic_io_semantics(state.pos_store));
   }

   if (writes_layer && layer_varying) {
      b.cursor = nir_after_impl(impl);
      nir_store_output(
         &b, nir_load_var(&b, state.layer), nir_imm_int(&b, 0),
         .base = util_bitcount64(nir->info.outputs_written &
                                 BITFIELD64_MASK(PAN_LAYER_CULL_SLOT)),
         .write_mask = 0x1, .component = 0, .src_type = nir_type_uint32,
         .io_semantics.location = PAN_LAYER_CULL_SLOT,
         .io_semantics.num_slots = 1);
      nir->info.outputs_written |= BITFIELD64_BIT(PAN_LAYER_CULL_SLOT);
   }

   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   nir->info.outputs_written &= ~(VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT);
   return true;
}

/* Layered draws: discard the fragments of primitives of other layers */
static bool
lower_fs_layer_cull(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   nir_def *layer = nir_load_input(
      &b, 1, 32, nir_imm_int(&b, 0),
      .base = util_bitcount64(nir->info.inputs_read &
                              BITFIELD64_MASK(PAN_LAYER_CULL_SLOT)),
      .component = 0, .dest_type = nir_type_uint32,
      .io_semantics.location = PAN_LAYER_CULL_SLOT,
      .io_semantics.num_slots = 1);
   nir_def *ref = nir_load_draw_layer_pan(&b);
   nir_demote_if(&b, nir_ine(&b, layer, ref));

   nir->info.inputs_read |= BITFIELD64_BIT(PAN_LAYER_CULL_SLOT);
   nir->info.fs.uses_discard = true;
   nir_progress(true, impl, nir_metadata_none);
   return true;
}

/* The fragment shader reads the layer drawn, not a varying (see above) */
static bool
lower_fs_layer_input(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input &&
       intr->intrinsic != nir_intrinsic_load_interpolated_input)
      return false;

   if (nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_LAYER)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *ref = nir_load_draw_layer_pan(b);
   nir_def *layer =
      nir_bcsel(b, nir_ieq_imm(b, ref, PAN_LAYER_ALL), nir_imm_int(b, 0), ref);
   layer = nir_replicate(b, nir_u2uN(b, layer, intr->def.bit_size),
                         intr->def.num_components);
   nir_def_replace(&intr->def, layer);
   return true;
}

/* CSO for a program derived from an application shader (libpoly lowering) */
static struct panfrost_uncompiled_shader *
panfrost_create_internal_shader(struct panfrost_context *ctx, nir_shader *nir)
{
   struct panfrost_device *dev = pan_device(ctx->base.screen);

   if (nir->info.stage == MESA_SHADER_VERTEX) {
      if (dev->arch < 9)
         lower_rast_layer_viewport(nir);

      pan_preprocess_nir(nir, dev->arch);
   }

   /* Programs built from scratch by libpoly have no default uniform block */
   nir->info.first_ubo_is_default_ubo = true;

   struct panfrost_uncompiled_shader *so = panfrost_alloc_shader(nir);
   ralloc_steal(so, nir);
   so->nir = nir;

   if (nir->info.stage == MESA_SHADER_VERTEX)
      panfrost_collect_vs_varying_layout(dev, so, nir);

   /* Precompile with the default key */
   struct panfrost_shader_key key = {0};
   panfrost_new_variant_locked(ctx, so, &key);
   return so;
}

static void panfrost_delete_shader_state(struct pipe_context *pctx, void *so);

static void
panfrost_create_gs_progs(struct panfrost_context *ctx,
                         struct panfrost_uncompiled_shader *so)
{
   struct panfrost_gs_progs *gs = rzalloc(so, struct panfrost_gs_progs);
   nir_shader *main_nir, *count_nir, *pre_gs_nir, *rast_nir;

   panfrost_nir_lower_gs(so->nir, &gs->info, &main_nir, &count_nir,
                         &pre_gs_nir, &rast_nir, &gs->wg_size,
                         &gs->vertex_stride);

   gs->main = panfrost_create_internal_shader(ctx, main_nir);
   gs->pre_gs = panfrost_create_internal_shader(ctx, pre_gs_nir);
   gs->xfb = gs->info.xfb;

   if (so->nir->xfb_info) {
      for (unsigned i = 0; i < POLY_MAX_SO_BUFFERS; ++i)
         gs->xfb_stride[i] = so->nir->xfb_info->buffers[i].stride;
   }

   gs->rast = panfrost_create_internal_shader(ctx, rast_nir);
   if (count_nir)
      gs->count = panfrost_create_internal_shader(ctx, count_nir);

   so->gs = gs;
}

static void
panfrost_create_tess_progs(struct panfrost_context *ctx,
                           struct panfrost_uncompiled_shader *so)
{
   const nir_shader *nir = so->nir;
   struct panfrost_tess_progs *t = rzalloc(so, struct panfrost_tess_progs);

   t->primitive = nir->info.tess._primitive_mode;
   t->spacing = nir->info.tess.spacing;
   t->ccw = nir->info.tess.ccw;
   t->point_mode = nir->info.tess.point_mode;
   t->output_patch_size = nir->info.tess.tcs_vertices_out;
   t->per_vertex_outputs = poly_tcs_per_vertex_outputs(nir);
   t->nr_patch_outputs = util_last_bit(nir->info.patch_outputs_written);

   if (nir->info.stage == MESA_SHADER_TESS_CTRL) {
      t->output_stride = poly_tcs_output_stride(nir);
      t->tcs = panfrost_create_internal_shader(ctx, panfrost_nir_lower_tcs(nir));
   } else {
      t->tes = panfrost_create_internal_shader(ctx, panfrost_nir_lower_tes(nir));
   }

   so->tess = t;
}

/* Keep the output variables and the transform feedback info of a vertex or
 * tessellation evaluation shader, to build passthrough geometry shaders.
 */
static nir_shader *
panfrost_keep_outputs_nir(void *mem_ctx, const nir_shader *nir)
{
   nir_shader *shell =
      nir_shader_create(mem_ctx, nir->info.stage, nir->options);

   shell->info = nir->info;
   shell->info.name = NULL;
   shell->info.label = NULL;

   /* Lowered I/O has no variables left: one per written slot, 4 components
    * (the geometry stages copy whole slots).
    */
   u_foreach_bit64(slot, nir->info.outputs_written) {
      const struct glsl_type *type = glsl_vec4_type();

      if (slot == VARYING_SLOT_PSIZ)
         type = glsl_float_type();
      else if (slot == VARYING_SLOT_LAYER || slot == VARYING_SLOT_VIEWPORT ||
               slot == VARYING_SLOT_PRIMITIVE_ID)
         type = glsl_int_type();

      nir_variable *var = nir_variable_create(shell, nir_var_shader_out, type,
                                              gl_varying_slot_name_for_stage(
                                                 slot, nir->info.stage));
      var->data.location = slot;
      var->data.driver_location = slot;
   }
   if (nir->xfb_info) {
      size_t size = nir_xfb_info_size(nir->xfb_info->output_count);
      shell->xfb_info = ralloc_memdup(shell, nir->xfb_info, size);
   }

   return shell;
}

static void *
panfrost_create_shader_state(struct pipe_context *pctx,
                             const struct pipe_shader_state *cso);

struct panfrost_uncompiled_shader *
panfrost_get_tes_compute(struct panfrost_context *ctx,
                         struct panfrost_uncompiled_shader *tes)
{
   struct panfrost_tess_progs *t = tes->tess;

   if (!t->tes_cs) {
      t->tes_cs = panfrost_create_internal_shader(
         ctx, panfrost_nir_lower_tes_to_compute(tes->nir));
   }

   return t->tes_cs;
}

struct panfrost_uncompiled_shader *
panfrost_get_passthrough_gs(struct panfrost_context *ctx,
                            struct panfrost_uncompiled_shader *prev,
                            enum mesa_prim mode)
{
   assert(mode < MESA_PRIM_COUNT && prev->outputs_nir);

   if (!prev->passthrough_gs[mode]) {
      nir_shader *gs = nir_create_passthrough_gs(
         ctx->base.screen->nir_options[MESA_SHADER_GEOMETRY],
         prev->outputs_nir, mode, mode, false, false, false);

      struct pipe_shader_state state = {
         .type = PIPE_SHADER_IR_NIR,
         .ir.nir = gs,
      };

      prev->passthrough_gs[mode] =
         panfrost_create_shader_state(&ctx->base, &state);
   }

   return prev->passthrough_gs[mode];
}

static void *
panfrost_create_shader_state(struct pipe_context *pctx,
                             const struct pipe_shader_state *cso)
{
   PAN_TRACE_FUNC(PAN_TRACE_GL_SHADER);

   nir_shader *nir = (cso->type == PIPE_SHADER_IR_TGSI)
                        ? tgsi_to_nir(cso->tokens, pctx->screen, false)
                        : cso->ir.nir;

   struct panfrost_uncompiled_shader *so = panfrost_alloc_shader(nir);

   /* The driver gets ownership of the nir_shader for graphics. The NIR is
    * ralloc'd. Free the NIR when we free the uncompiled shader.
    */
   ralloc_steal(so, nir);

   so->stream_output = cso->stream_output;
   so->nir = nir;

   if ((nir->info.stage == MESA_SHADER_VERTEX ||
        nir->info.stage == MESA_SHADER_TESS_EVAL) &&
       pan_device(pctx->screen)->arch >= 6 &&
       pan_device(pctx->screen)->arch <= 7)
      so->outputs_nir = panfrost_keep_outputs_nir(so, nir);

   /* PLS lowering is not taken care of by glsl_to_nir(), so do it here. */
   if (nir->info.stage == MESA_SHADER_FRAGMENT &&
       nir->info.fs.accesses_pixel_local_storage) {
      /* Try to optimize the case where inout PLS vars are never
       * read/written to. Needs to be called before
       * nir_lower_io_vars_to_temporaries() because the copy_derefs
       * inserted there prevent us from detecting PLS usage.
       */
      NIR_PASS(_, nir, nir_downgrade_pls_vars);

      /* Lower PLS vars to temporaries before we lower IOs. */
      NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries,
               nir_shader_get_entrypoint(nir), nir_var_any_pixel_local);

      /* We need to lower all the copy_deref's introduced by lower_io_to-
       * _temporaries before calling nir_lower_io.
       */
      NIR_PASS(_, nir, nir_split_var_copies);
      NIR_PASS(_, nir, nir_lower_var_copies);
      NIR_PASS(_, nir, nir_lower_global_vars_to_local);

      /* Lower all PLS IOs. */
      NIR_PASS(_, nir, nir_lower_io, nir_var_any_pixel_local, glsl_type_size,
               0);

      /* Lower and remove dead derefs and variables to clean up the IR. */
      NIR_PASS(_, nir, nir_lower_vars_to_ssa);
      NIR_PASS(_, nir, nir_opt_dce);
      NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_function_temp, NULL);

      /* Re-run gather_info() to get the latest accesses_pixel_local_storage
       * state.
       */
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   }

   struct panfrost_device *dev = pan_device(pctx->screen);

   if (nir->info.stage == MESA_SHADER_FRAGMENT &&
       nir->info.outputs_written & BITFIELD_BIT(FRAG_RESULT_COLOR)) {
      /* Bifrost and newer handle gl_FragColor in the backend, but Midgard
       * still needs the frontend lowering.  Must happen before IO lowering
       */
      if (dev->arch <= 5) {
         assert(!nir->info.io_lowered);
         NIR_PASS(_, nir, nir_lower_fragcolor,
                  nir->info.fs.color_is_dual_source ? 1 : 8);
      }
      so->fragcolor_lowered = true;
   }

   /* Then run the suite of lowering and optimization, including I/O lowering */
   pan_preprocess_nir(nir, dev->arch);

   /* Usually IO is lowered by gallium, but TTN doesn't */
   if (!nir->info.io_lowered) {
      NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
               nir_var_shader_in | nir_var_shader_out, UINT32_MAX);
      NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
               glsl_type_size, nir_lower_io_use_interpolated_input_intrinsics);
      /* nir_lower_io just computes offsets based on the original deref and
       * lower_indirect_derefs ensures that the array derefs have a constant
       * index.  Constant-fold to get us actual constants in in load/store
       * instructions.
       */
      NIR_PASS(_, nir, nir_opt_constant_folding);
   }

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      if (dev->arch >= 6 && dev->arch <= 7 &&
          (nir->info.inputs_read & VARYING_BIT_LAYER)) {
         NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_fs_layer_input,
                  nir_metadata_control_flow, NULL);
         nir->info.inputs_read &= ~VARYING_BIT_LAYER;
      }

      so->noperspective_varyings =
         pan_nir_collect_noperspective_varyings_fs(nir);

      NIR_PASS(_, nir, nir_shader_intrinsics_pass,
               lower_sample_mask_writes, nir_metadata_control_flow, NULL);

      if (nir->info.fs.accesses_pixel_local_storage)
         NIR_PASS(_, nir, panfrost_nir_lower_pls, pan_screen(pctx->screen));
   }

   if (nir->info.stage == MESA_SHADER_VERTEX) {
      panfrost_collect_vs_varying_layout(dev, so, nir);
      so->outputs_written = nir->info.outputs_written;
   }

   struct panfrost_context *ctx = pan_context(pctx);

   /* Geometry shaders are lowered to several programs, compiled now */
   if (nir->info.stage == MESA_SHADER_GEOMETRY) {
      panfrost_create_gs_progs(ctx, so);
      return so;
   }

   if (nir->info.stage == MESA_SHADER_TESS_CTRL ||
       nir->info.stage == MESA_SHADER_TESS_EVAL) {
      panfrost_create_tess_progs(ctx, so);
      return so;
   }

   /* If this shader uses transform feedback, compile the transform
    * feedback program. This is a special shader variant.
    */

   if (so->nir->xfb_info) {
      so->xfb = calloc(1, sizeof(struct panfrost_compiled_shader));
      so->xfb->key.vs.is_xfb = true;

      panfrost_shader_get(ctx->base.screen, &ctx->shaders, &ctx->descs, so,
                          &ctx->base.debug, so->xfb, 0);

      /* Since transform feedback is handled via the transform
       * feedback program, the original program no longer uses XFB
       */
      pan_nir_remove_xfb(nir);
   }

   /* If we're not using separate shaders, the FS can use VS varying_layout to
    * optimize loads (LD_VAR_BUF instead of LD_VAR).  Gallium won't provide us
    * with the VS directly, so we need to delay the default variant compilation
    * until link time
    */
   if (nir->info.stage == MESA_SHADER_FRAGMENT && !nir->info.separate_shader)
      return so;

   /* Compile the program. We don't use vertex shader keys, so there will
    * be no further vertex shader variants. We do have fragment shader
    * keys, but we can still compile with a default key that will work most
    * of the time.
    */
   struct panfrost_shader_key key = panfrost_default_shader_key(so);

   /* Creating a CSO is single-threaded, so it's ok to use the
    * locked function without explicitly taking the lock. Creating a
    * default variant acts as a precompile.
    */
   panfrost_new_variant_locked(ctx, so, &key);

   return so;
}

static void
panfrost_delete_shader_state(struct pipe_context *pctx, void *so)
{
   struct panfrost_uncompiled_shader *cso =
      (struct panfrost_uncompiled_shader *)so;

   util_dynarray_foreach(&cso->variants, struct panfrost_compiled_shader, so) {
      panfrost_bo_unreference(so->bin.bo);
      panfrost_bo_unreference(so->state.bo);
      panfrost_bo_unreference(so->linkage.bo);
   }

   if (cso->xfb) {
      panfrost_bo_unreference(cso->xfb->bin.bo);
      panfrost_bo_unreference(cso->xfb->state.bo);
      panfrost_bo_unreference(cso->xfb->linkage.bo);
      free(cso->xfb);
   }

   if (cso->gs) {
      panfrost_delete_shader_state(pctx, cso->gs->main);
      panfrost_delete_shader_state(pctx, cso->gs->rast);
      if (cso->gs->count)
         panfrost_delete_shader_state(pctx, cso->gs->count);
      panfrost_delete_shader_state(pctx, cso->gs->pre_gs);
   }

   for (unsigned i = 0; i < ARRAY_SIZE(cso->passthrough_gs); ++i) {
      if (cso->passthrough_gs[i])
         panfrost_delete_shader_state(pctx, cso->passthrough_gs[i]);
   }

   if (cso->tess) {
      if (cso->tess->tcs)
         panfrost_delete_shader_state(pctx, cso->tess->tcs);
      if (cso->tess->tes)
         panfrost_delete_shader_state(pctx, cso->tess->tes);
      if (cso->tess->tes_cs)
         panfrost_delete_shader_state(pctx, cso->tess->tes_cs);
   }

   simple_mtx_destroy(&cso->lock);

   ralloc_free(so);
}

static void
panfrost_link_shader(struct pipe_context *pctx, void** handles)
{
   struct panfrost_context *ctx = pan_context(pctx);
   struct panfrost_uncompiled_shader *vs = handles[MESA_SHADER_VERTEX];
   struct panfrost_uncompiled_shader *fs = handles[MESA_SHADER_FRAGMENT];

   if (!fs || fs->nir->info.separate_shader)
      return;

   struct panfrost_uncompiled_shader *gs = handles[MESA_SHADER_GEOMETRY];
   struct panfrost_uncompiled_shader *tes = handles[MESA_SHADER_TESS_EVAL];

   /* The FS is linked with the hardware vertex shader of the rasterization
    * draw: the geometry shader's rasterization program, else the tessellation
    * evaluation shader, else the vertex shader. Sometimes there is no VS
    * (e.g. fixed-function VS), in those cases we just compile a default FS.
    */
   if (gs && gs->gs)
      fs->vs_varying_layout = gs->gs->rast->vs_varying_layout;
   else if (tes && tes->tess)
      fs->vs_varying_layout = tes->tess->tes->vs_varying_layout;
   else if (vs)
      fs->vs_varying_layout = vs->vs_varying_layout;

   simple_mtx_lock(&fs->lock);

   struct panfrost_shader_key key = panfrost_default_shader_key(fs);
   panfrost_new_variant_locked(ctx, fs, &key);

   simple_mtx_unlock(&fs->lock);
}

/*
 * Create a compute CSO. As compute kernels do not require variants, they are
 * precompiled, creating both the uncompiled and compiled shaders now.
 */
static void *
panfrost_create_compute_state(struct pipe_context *pctx,
                              const struct pipe_compute_state *cso)
{
   struct panfrost_context *ctx = pan_context(pctx);
   struct panfrost_uncompiled_shader *so = panfrost_alloc_shader(cso->prog);
   struct panfrost_compiled_shader *v = panfrost_alloc_variant(so);
   memset(v, 0, sizeof *v);

   assert(cso->ir_type == PIPE_SHADER_IR_NIR && "TGSI kernels unsupported");

   panfrost_shader_get(pctx->screen, &ctx->shaders, &ctx->descs, so,
                       &ctx->base.debug, v, cso->static_shared_mem);

   /* The NIR becomes invalid after this. For compute kernels, we never
    * need to access it again. Don't keep a dangling pointer around.
    */
   ralloc_free((void *)so->nir);
   so->nir = NULL;

   return so;
}

static void
panfrost_bind_compute_state(struct pipe_context *pipe, void *cso)
{
   struct panfrost_context *ctx = pan_context(pipe);
   struct panfrost_uncompiled_shader *uncompiled = cso;

   ctx->uncompiled[MESA_SHADER_COMPUTE] = uncompiled;

   ctx->prog[MESA_SHADER_COMPUTE] =
      uncompiled ? util_dynarray_begin(&uncompiled->variants) : NULL;
}

static void
panfrost_get_compute_state_info(struct pipe_context *pipe, void *cso,
                                struct pipe_compute_state_object_info *info)
{
   struct panfrost_device *dev = pan_device(pipe->screen);
   struct panfrost_uncompiled_shader *uncompiled = cso;
   struct panfrost_compiled_shader *cs =
      util_dynarray_begin(&uncompiled->variants);

   info->max_threads = pan_compute_max_thread_count(&dev->kmod.dev->props,
                                                    cs->info.work_reg_count);
   info->private_memory = cs->info.tls_size;
   info->simd_sizes = pan_subgroup_size_gpu(panfrost_device_gpu_id(dev));
   info->preferred_simd_size = info->simd_sizes;
}

void
panfrost_shader_context_init(struct pipe_context *pctx)
{
   pctx->create_vs_state = panfrost_create_shader_state;
   pctx->delete_vs_state = panfrost_delete_shader_state;
   pctx->bind_vs_state = panfrost_bind_vs_state;

   pctx->create_tcs_state = panfrost_create_shader_state;
   pctx->delete_tcs_state = panfrost_delete_shader_state;
   pctx->bind_tcs_state = panfrost_bind_tcs_state;

   pctx->create_tes_state = panfrost_create_shader_state;
   pctx->delete_tes_state = panfrost_delete_shader_state;
   pctx->bind_tes_state = panfrost_bind_tes_state;

   pctx->create_gs_state = panfrost_create_shader_state;
   pctx->delete_gs_state = panfrost_delete_shader_state;
   pctx->bind_gs_state = panfrost_bind_gs_state;

   pctx->create_fs_state = panfrost_create_shader_state;
   pctx->delete_fs_state = panfrost_delete_shader_state;
   pctx->bind_fs_state = panfrost_bind_fs_state;

   pctx->link_shader = panfrost_link_shader;

   pctx->create_compute_state = panfrost_create_compute_state;
   pctx->bind_compute_state = panfrost_bind_compute_state;
   pctx->get_compute_state_info = panfrost_get_compute_state_info;
   pctx->delete_compute_state = panfrost_delete_shader_state;
}
