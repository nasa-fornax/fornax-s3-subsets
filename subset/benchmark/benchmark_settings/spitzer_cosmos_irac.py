"""
test parameters for the big IRAC COSMOS mosaics. There are only 12 of these 
mosaics, and this is all of them. single f32 image HDU, uncompressed. ~480 MB.
"""
CUT_SHAPES = ((40, 40), (200, 200))
CUT_COUNTS = (1, 5, 20)
BUCKET = "nishapur"
AUTHENTICATE_S3 = True
HDU_IX = 0
LOADERS = (
    "astropy", "fitsio", "astropy_s3_section", "astropy_s3", "greedy_astropy"
)
TEST_FILES = (
    "spitzer/cosmos/irac_ch1_go2_cov_10.fits",
    "spitzer/cosmos/irac_ch1_go2_unc_10.fits",
    "spitzer/cosmos/irac_ch2_go2_sci_10.fits",
    "spitzer/cosmos/irac_ch3_go2_cov_10.fits",
    "spitzer/cosmos/irac_ch3_go2_unc_10.fits",
    "spitzer/cosmos/irac_ch4_go2_sci_10.fits",
    "spitzer/cosmos/irac_ch1_go2_sci_10.fits",
    "spitzer/cosmos/irac_ch2_go2_cov_10.fits",
    "spitzer/cosmos/irac_ch2_go2_unc_10.fits",
    "spitzer/cosmos/irac_ch3_go2_sci_10.fits",
    "spitzer/cosmos/irac_ch4_go2_cov_10.fits",
    "spitzer/cosmos/irac_ch4_go2_unc_10.fits",
)
