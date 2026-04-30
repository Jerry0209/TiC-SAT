#include <SVE_implementations.h>


svfloat32_t extract_weightsx2(svbool_t pg, svuint32_t idxs, svfloat32x2_t cb_resg_x2){

    svbool_t cb_pred = svcmplt_u32(pg, idxs, svdup_u32(4));
    svfloat32_t w_lo = svdup_f32(0.0);
    w_lo = svtbl_f32(svget2_f32(cb_resg_x2, 0), idxs);

    // printf("WLO\n");
    // // print_vect_f32(w_lo);
    // print_vect_f32(svget2_f32(cb_resg_x2, 0));

    cb_pred = svnot_b_z(pg, cb_pred);
    svuint32_t idx_hi = svsub_n_u32_z(cb_pred, idxs, 4);
    svfloat32_t w_hi = svtbl_f32(svget2_f32(cb_resg_x2, 1), idx_hi);

    return svsel_f32(cb_pred, w_hi, w_lo);
}




svfloat32_t extract_weightsx4(svbool_t pg, svuint32_t idxs, svfloat32x4_t cb_regs_x4){

    svbool_t cb_pred = svcmplt_u32(pg, idxs, svdup_u32(4));

    svfloat32_t w0 = svdup_f32(0.0);
    w0 = svtbl_f32(svget4_f32(cb_regs_x4, 0), idxs);
    
    cb_pred = svcmplt_u32(pg, idxs, svdup_u32(8));
    cb_pred = svand_z(pg, cb_pred, svnot_z(pg, svcmplt_u32(pg, idxs, svdup_u32(4))));
    svuint32_t idxs1 = svsub_n_u32_z(cb_pred, idxs, 4);
    svfloat32_t w1 = svtbl_f32(svget4_f32(cb_regs_x4, 1), idxs1);

    svfloat32_t res = svsel_f32(cb_pred, w1, w0);
    // print_vect_f32(res);

    cb_pred = svcmplt_u32(pg, idxs, svdup_u32(12));
    cb_pred = svand_z(pg, cb_pred, svnot_z(pg, svcmplt_u32(pg, idxs, svdup_u32(8))));
    svuint32_t idxs2 = svsub_n_u32_z(cb_pred, idxs, 8);
    svfloat32_t w2 = svtbl_f32(svget4_f32(cb_regs_x4, 2), idxs2);
    
    res = svsel_f32(cb_pred, w2, res);
    
    cb_pred = svcmpge_u32(pg, idxs, svdup_u32(12));
    svuint32_t idxs3 = svsub_n_u32_z(cb_pred, idxs, 12);
    svfloat32_t w3 = svtbl_f32(svget4_f32(cb_regs_x4, 3), idxs3);

    return svsel_f32(cb_pred, w3, res);
}