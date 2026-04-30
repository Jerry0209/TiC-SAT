#include <SVE_implementations.h>


/**
 * This funcion retrives fp32 weights from an svfloat16_t vector register at the positions indicated by idxs.
 */
svfloat32_t get_weights_f32(svfloat16_t codebook_f16, svuint32_t idxs){

    // printf("============\n");
    // print_vect_ui32(idxs);
    // print_vect_f16(codebook_f16);

    svuint16_t idxs16 = svreinterpret_u16_u32(idxs);
    // print_vect_ui16(idxs16);    

    svfloat16_t vals = svtbl_f16(codebook_f16, idxs16);
    // print_vect_f16(vals);

    svbool_t conv_pg = svptrue_b16();
    return svcvt_f32_f16_z(conv_pg, vals);
}


svfloat16_t extract_weightsx2_f16(svbool_t pg, svuint16_t idxs, svfloat16x2_t cb_resg_x2){

    svbool_t cb_pred = svcmplt_u16(pg, idxs, svdup_u16(8));
    svfloat16_t w_lo = svdup_f16(0.0);
    w_lo = svtbl_f16(svget2_f16(cb_resg_x2, 0), idxs);

    cb_pred = svnot_b_z(pg, cb_pred);
    svuint16_t idx_hi = svsub_n_u16_z(cb_pred, idxs, 8);
    svfloat16_t w_hi = svtbl_f16(svget2_f16(cb_resg_x2, 1), idx_hi);

    return svsel_f16(cb_pred, w_hi, w_lo);
}


// svfloat16_t extract_weightsx2_f16(svbool_t pg, svuint16_t idxs, svfloat16x2_t cb_resg_x2){

//     svbool_t cb_pred = svcmplt_u32(pg, idxs, svdup_u32(8));
//     // print_predicate_h(cb_pred);

//     // Weights taken from the lower half of the codebook (aka index lower than 8)
//     svfloat16_t w_lo_f16 = svdup_f16(0.0);
//     w_lo_f16 = svtbl_f16(svget2_f16(cb_resg_x2, 0), svreinterpret_u16_u32(idxs));
//     svfloat32_t w_lo_f32 = svcvt_f32_f16_x(pg, w_lo_f16);
//     // print_vect_f32(w_lo_f32);


//     cb_pred = svnot_b_z(pg, cb_pred);
//     svuint32_t idx_hi = svsub_n_u32_z(cb_pred, idxs, 8);
//     svfloat16_t w_hi_f16 = svtbl_f16(svget2_f16(cb_resg_x2, 1), svreinterpret_u16_u32(idx_hi));
//     svfloat32_t w_hi_f32 = svcvt_f32_f16_x(pg, w_hi_f16);
//     // print_vect_f32(w_hi_f32);

//     exit(0);

//     return svsel_f32(cb_pred, w_hi_f32, w_lo_f32);
// }