/*
 * Copyright 2026 Arun Iyengar
 * SPDX-License-Identifier: MIT
 *
 * Tessellation on job manager Bifrost: the libpoly tessellator, and the job
 * patching that sizes the rasterization draw on the GPU.
 */

#include "compiler/libcl/libcl.h"
#include "genxml/gen_macros.h"
#include "lib/pan_encoder.h"

#if (PAN_ARCH >= 6 && PAN_ARCH <= 7)
#include "poly/cl/tessellator.h"
#include "poly/geometry.h"

/*
 * Geometry shaders with transform feedback and unknown output counts: inclusive
 * prefix sum of each word of the per-primitive counts written by the count
 * program. Serial: one invocation.
 */
KERNEL(1)
panlib_gs_prefix_sum(global struct poly_geometry_params *p)
{
   uint words = p->count_buffer_stride / 4;

   for (uint w = 0; w < words; ++w) {
      uint sum = 0;

      for (uint i = 0; i < p->input_primitives; ++i) {
         sum += p->count_buffer[i * words + w];
         p->count_buffer[i * words + w] = sum;
      }
   }
}

/* One invocation per patch. prim is a enum tess_primitive_mode. */
KERNEL(64)
panlib_tessellate(constant struct poly_tess_params *p, uint32_t prim,
                  uint32_t mode)
{
   uint patch = cl_global_id.x;
   if (patch >= p->nr_patches)
      return;

   enum poly_tess_mode m = (enum poly_tess_mode)mode;

   if (prim == TESS_PRIMITIVE_ISOLINES)
      poly_tess_isoline_process(p, patch, m);
   else if (prim == TESS_PRIMITIVE_QUADS)
      poly_tess_quad_process(p, patch, m);
   else
      poly_tess_tri_process(p, patch, m);
}

/*
 * After the COUNT pass: prefix sum the per-patch index and domain point
 * counts, then write the vertex count (domain points) and the index count
 * into the vertex and tiler jobs of the rasterization draw, which follow in
 * the job chain. jobs[] holds their GPU addresses (written by the CPU after
 * this job was queued). Serial: one invocation.
 */
KERNEL(1)
panlib_tess_prefix_sum_patch(global struct poly_tess_params *p,
                             constant uint64_t *jobs)
{
   uint indices = 0, points = 0;

   for (uint i = 0; i < p->nr_patches; ++i) {
      indices += p->counts[i];
      p->counts[i] = indices;

      points += p->point_counts[i];
      p->point_counts[i] = points;
   }

   /* An empty draw still needs one vertex and one index: the first index is
    * the restart index then, so nothing is rasterized.
    */
   uint vertex_count = max(points, 1u);
   uint index_count = max(indices, 1u);

   /* No jobs: the CPU reads the counts back and sizes the draw */
   if (!jobs[0])
      return;

   if (indices == 0)
      p->index_buffer[0] = ~0u;

   struct mali_invocation_packed invocation;
   pan_pack(&invocation, INVOCATION, cfg) {
      cfg.invocations = vertex_count - 1;
      cfg.size_y_shift = 0;
      cfg.size_z_shift = 0;
      cfg.workgroups_x_shift = 0;
      cfg.workgroups_y_shift = 0;
      cfg.workgroups_z_shift = 32;
      cfg.thread_group_split = MALI_SPLIT_MIN_EFFICIENT;
   }

   global uint8_t *vertex_job = (global uint8_t *)jobs[0];
   global uint8_t *tiler_job = (global uint8_t *)jobs[1];

   *((global struct mali_invocation_packed *)(vertex_job +
      pan_section_offset(COMPUTE_JOB, INVOCATION))) = invocation;
   *((global struct mali_invocation_packed *)(tiler_job +
      pan_section_offset(TILER_JOB, INVOCATION))) = invocation;

   global struct mali_primitive_packed *primitive =
      (global struct mali_primitive_packed *)(tiler_job +
         pan_section_offset(TILER_JOB, PRIMITIVE));

   pan_unpack(primitive, PRIMITIVE, unpacked);
   pan_pack(primitive, PRIMITIVE, cfg) {
      memcpy(&cfg, &unpacked, sizeof(cfg));
      cfg.index_count = index_count;
   }
}
#endif
