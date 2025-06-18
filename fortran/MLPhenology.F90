program MLPhenology

  use               FatesConstantsMod, only : r8 => fates_r8
  use              FatesArgumentUtils, only : command_line_arg
  use             FatesUnitTestIOMod,  only : OpenNCFile, GetVar, CloseNCFile, RegisterNCDims

  implicit none

  ! define ML phenoogy pytorch model
  character(len=256) :: the_torch_model = "/glade/u/home/linnia/MLphenology/models/example_LSTM_model_lh.pt"
  character(len=256) :: the_tft_torch_model = "/glade/u/home/ayal/phenology-ml-clm/models/tft_scripted.pt"
  !character(len=256) :: the_torch_model = "/glade/u/home/ayal/phenology-ml-clm/models/example_LSTM_model_v1.pt"
  
  real(8), dimension(10) :: dummy_lai
  integer :: sos_flag, n

  character(len=:),                  allocatable :: datm_file            ! input DATM 
  real(r8),                          allocatable :: ta(:)         ! daily air temperature [degC]
  real(r8),                          allocatable :: pr(:)            ! daily precipitation [mm]
  real(r8),                          allocatable :: sw(:)                ! daily shortwave radiation (W/m2)
  real(r8),                          allocatable :: lai(:)              ! daily LAI (m2/m2)

  real(r8)                                       :: doy_arr(10)              ! DOY array
  real(r8)                                       :: out_data(1,5)       ! output from the lstm model (lai)
  real(r8)                                       :: out_data_tft(1,10)       ! output from the tft model (lai)

  real(r8)                                       :: soilt            ! soil temperature at 12cm
  real(r8)                                       :: doy ! day of year (used to identify solstace) 
  real(r8)                                       :: onset_gdd      ! onset growing degree days 
  real(r8)                                       :: onset_gddflag  ! Onset freeze flag

  logical                                        :: do_onset       ! Flag if onset should happen

  
  ! Load forcing data
  datm_file = command_line_arg(1) ! one year of daily ta, pr, sw, lai
  call load_met_forcing(datm_file, ta, pr, sw, lai)

  ! ======================================
  ! test CLM SeasonalDecidOnset function
  doy = 1.0_r8
  onset_gdd = 0.0_r8
  onset_gddflag = 1.0_r8

  soilt = ta(doy)-10.0_r8

  do_onset = SeasonalDecidOnset( onset_gdd, onset_gddflag, soilt, doy )
  print *, "onset_gdd: ", onset_gdd
  print *, "onset_gddflag: ", onset_gddflag

  ! ====================================
  ! test SOS
  n = 10
  dummy_lai = (/1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0/)
  call get_sos(dummy_lai, n, sos_flag)
  print *, "start of season Flag:", sos_flag


  ! ========================================
  ! test tft

  call next_ten_days(doy, doy_arr)
  call run_tft_model(the_tft_torch_model, ta, pr, sw, lai, doy_arr, out_data_tft)
  print *, "TFT predicted LAI:", out_data_tft

  contains
  
    !-----------------------------------------------------------------------
    subroutine load_met_forcing ( datm_file, ta, pr, sw, lai)
      ! 
      use FatesConstantsMod, only: r8 => fates_r8
      use FatesUnitTestIOMod, only: OpenNCFile, GetVar, CloseNCFile
    
      implicit none
    
      ! Arguments
      character(len=*), intent(in) :: datm_file
      real(r8), allocatable, intent(out) :: ta(:), pr(:), sw(:), lai(:)
    
      ! Local
      integer :: ncid
    
      ! Allocate arrays
      allocate(ta(365), pr(365), sw(365), lai(365))
    
      ! Open and read
      call OpenNCFile(trim(datm_file), ncid, 'read')
      
      call GetVar(ncid, 'ta', ta)
      call GetVar(ncid, 'pr', pr)
      call GetVar(ncid, 'sw', sw)
      call GetVar(ncid, 'lai', lai)
    
      call CloseNCFile(ncid)
    
    end subroutine load_met_forcing
    
    ! ----------------------------------------------------------------------

    subroutine get_sos(lai, n, sos_flag)
        real(8), dimension(n), intent(in) :: lai
        integer, intent(in) :: n
        integer, intent(out) :: sos_flag
        real(8), dimension(n) :: t, kv
        real(8) :: a, b, c, d, max_annual_lai, annual_lai_amp
        real(8) :: z, z1, z2, z3, p1n, p1d, p2n, p2d
        integer :: i
      
        ! Initialize variables
        max_annual_lai = 1
        annual_lai_amp = 0.5
        do i = 1, n
          t(i) = real(i - 1, 8)
        end do
      
        ! Initial parameter guesses
        a = -5.0
        b = 0.1
        c = maxval(lai)
        d = minval(lai)
      
        ! Calculate curvature rate of change
        do i = 1, size(t)
            z = exp(a + b * t(i))
            z1 = 1.0 - z
            z2 = 1.0 + z
            z3 = (b * c * z)**2
            p1n = 3.0 * z * z1 * z2**3 * (2.0 * z2**3 + b**2 * c**2 * z)
            p1d = (z2**4 + z3)**(2.5)
            p2n = z2**2 * (1.0 + 2.0 * z - 5.0 * z**2)
            p2d = (z2**4 + z3)**(1.5)
            kv(i) = b**3 * c * z * ((p1n / p1d) - (p2n / p2d))
          end do
      
        if (maxval(lai) > max_annual_lai * 0.3 .and. maxval(lai) - minval(lai) > 0.3 * annual_lai_amp) then
          sos_flag = 0
          do i = 2, n - 1
            if (kv(i) > kv(i - 1) .and. kv(i) > kv(i + 1)) then
                sos_flag = 1
                return
            end if
          end do
        else
            sos_flag = 0
        end if
      
      
    end subroutine get_sos

  !-----------------------------------------------------------------------
    function SeasonalDecidOnset( onset_gdd, onset_gddflag, soilt, doy ) &
                       result( do_onset )

        ! !DESCRIPTION:
        ! Function to determine if seasonal deciduous leaf onset should happen.
        !
        ! !ARGUMENTS:
        real(r8), intent(INOUT) :: onset_gdd      ! onset growing degree days 
        real(r8), intent(INOUT) :: onset_gddflag  ! Onset freeze flag
        real(r8), intent(IN)    :: soilt          ! Soil temperature at specific level for this evaluation
        real(r8), intent(IN)    :: doy            ! day of year
        logical :: do_onset                       ! Flag if onset should happen (return value)

        ! !LOCAL VARIABLES:
        real(r8):: ws_flag        !winter-summer solstice flag (0 or 1)
        real(r8):: crit_onset_gdd !critical onset growing degree-day sum
        real(r8):: crit_dayl      ! parameter
        real(r8):: annavg_t2m_patch        

        !-----------------------------------------------------------------------
        ! set constants
        annavg_t2m_patch = 15 ! annual average patch temperature (C)
        crit_dayl = 39300 ! seconds
        
        ! onset gdd sum from Biome-BGC, v4.1.2
        crit_onset_gdd = exp(4.8_r8 + 0.13_r8*(annavg_t2m_patch))
    
        ! set flag for solstice period (winter->summer = 1, summer->winter = 0)
        if (doy <= 171) then
          ws_flag = 1._r8
        else
          ws_flag = 0._r8
        end if
    
        do_onset = .false.
        ! Test to turn on growing degree-day sum, if off.
        ! switch on the growing degree day sum on the winter solstice
    
        if (onset_gddflag == 0._r8 .and. ws_flag == 1._r8) then
            onset_gddflag = 1._r8
            onset_gdd = 0._r8
        end if
    
        ! Test to turn off growing degree-day sum, if on.
        ! This test resets the growing degree day sum if it gets past
        ! the summer solstice without reaching the threshold value.
        ! In that case, it will take until the next winter solstice
        ! before the growing degree-day summation starts again.
    
        if (onset_gddflag == 1._r8 .and. ws_flag == 0._r8) then
            onset_gddflag = 0._r8
            onset_gdd = 0._r8
        end if
    
        ! if the gdd flag is set, and if the soil is above freezing
        ! then accumulate growing degree days for onset trigger
    
        if (onset_gddflag == 1.0_r8 .and. soilt > 273.15_r8) then
            onset_gdd = onset_gdd + (soilt-273.15_r8)
        end if

        ! set do_onset if critical growing degree-day sum is exceeded
        if (onset_gdd > crit_onset_gdd) then
            do_onset = .true.
        end if
    
    end function SeasonalDecidOnset

    !-----------------------------------------------------------------------


    subroutine next_ten_days(doy, doy_arr)
      real(r8), intent(in)  :: doy
      real(r8), intent(out) :: doy_arr(10)
      integer :: i
      real(r8) :: mod_doy
      mod_doy = real(365, r8)

      do i = 1, 10
        doy_arr(i) = mod(doy + real(i, r8) - real(1, r8), mod_doy) + real(1, r8)
      end do
    end subroutine next_ten_days

    subroutine run_tft_model (the_tft_torch_model, ta, pr, sw, lai, doy_arr, out_data_tft)

        use   iso_c_binding,     only : c_float, c_int
        use   ftorch,            only : torch_model, torch_model_load, torch_model_forward, &
                                        torch_tensor, torch_tensor_from_array, torch_kCPU,  torch_delete
        implicit none
    
        ! Arguments
        character(len=*), intent(in) :: the_tft_torch_model
        real(r8),         intent(in) :: ta(:), pr(:), sw(:), lai(:), doy_arr(10)
        real(r8),         intent(out) :: out_data_tft(1,10)

        ! Local
        type(torch_model)                             :: model_pytorch
        type(torch_tensor), allocatable               :: inputs(:)
        type(torch_tensor), allocatable               :: outputs(:)
        real(c_float),     dimension(0)               :: empty_cat_raw
        real(c_float),      dimension(1,2)            :: static_num                           ! latitude, longitude
        integer(c_int),     dimension(1,0)            :: static_cat                           ! no static categorical features
        real(c_float),      dimension(1,60,8)         :: hist_num                             ! tmin, tmax, precip, rad, photoperiod, swvl1, doy, lai
        integer(c_int),     dimension(1,60,0)         :: hist_cat                             ! no historical categorical features
        real(c_float),      dimension(1,10,1)         :: fut_num                              ! doy
        integer(c_int),     dimension(1,10,0)         :: fut_cat                              ! no future categorical features
        real(c_float),      dimension(1,10,3)         :: out_quantiles                           ! output quantiles (0.1, 0.5, 0.9)
        integer(c_int),     allocatable               :: L2(:), L3(:)
        integer                                       :: n_in, n_out, i
        real(c_float), dimension(8) :: hmin, hmax
        real(c_float)              :: fmin, fmax
        integer                    :: j

        n_in  = 6                                                                             ! 6 input tensors
        n_out = 1                                                                             ! 1 output tensor (the 3‐quantile forecast)

        !---  Populate input data (first n_input days)
        static_num= reshape([ 1_c_float, 1_c_float ], [1,2])    ! TD: substitute with true lat/lon

        hist_num(1,:,1)= real(ta(1:60), c_float)                        ! TD: replace with tmin
        hist_num(1,:,2)= real(ta(1:60), c_float)                        ! TD: replace with tmax 
        hist_num(1,:,3)= real(pr(1:60), c_float)
        hist_num(1,:,4)= real(sw(1:60), c_float)
        ! TODO: Replace the following placeholder assignments with the correct variables for each feature
        hist_num(1,:,5)= real(ta(1:60), c_float)                        ! TODO: replace with photoperiod variable
        hist_num(1,:,6)= real(ta(1:60), c_float)                        ! TODO: replace with soil moisture variable
        hist_num(1,:,7)= real(ta(1:60), c_float)                        ! TODO: replace with day of year variable
        hist_num(1,:,8)= real(ta(1:60), c_float)                        ! TODO: replace with lai variable

        fut_num(1,:,1) = real(doy_arr(1:10), c_float)                   ! future 10 days of year

        static_cat    = reshape(empty_cat_raw, [1,0])                   ! no static categorical features
        hist_cat      = reshape(empty_cat_raw, [1,60,0])                ! no historical categorical features
        fut_cat       = reshape(empty_cat_raw, [1,10,0])                ! no future categorical features

        ! ---- normalize historical channels ----
        do j = 1, 8
          hmin(j) = minval( hist_num(1, :, j) )
          hmax(j) = maxval( hist_num(1, :, j) )
          if (hmax(j) > hmin(j)) then
            hist_num(1, :, j) = ( hist_num(1, :, j) - hmin(j) ) / (hmax(j) - hmin(j))
          else
            hist_num(1, :, j) = 0.0_c_float  ! or leave at 0 if flat
          end if
        end do

        ! ---- normalize future single channel ----
        fmin = minval( fut_num(1, :, 1) )
        fmax = maxval( fut_num(1, :, 1) )
        if (fmax > fmin) then
          fut_num(1, :, 1) = ( fut_num(1, :, 1) - fmin ) / (fmax - fmin)
        else
          fut_num(1, :, 1) = 0.0_c_float
        end if

        !===============
        ! load pytorch model
        call torch_model_load(model_pytorch, trim(the_tft_torch_model), torch_kCPU)

        !===============
        ! Allocate arrays of tensor handles
        allocate(inputs(n_in))
        allocate(outputs(n_out))

        allocate(L2(2));  L2  = [1_c_int,2_c_int]
        allocate(L3(3)); L3 = [1_c_int,2_c_int,3_c_int]
                
        !===============
        ! Wrap each Fortran array in a torch_tensor
        call torch_tensor_from_array(inputs(1), static_num, L2, torch_kCPU)
        call torch_tensor_from_array(inputs(2), static_cat, L2, torch_kCPU)
        call torch_tensor_from_array(inputs(3), hist_num,   L3, torch_kCPU)
        call torch_tensor_from_array(inputs(4), hist_cat,   L3, torch_kCPU)
        call torch_tensor_from_array(inputs(5), fut_num,    L3, torch_kCPU)
        call torch_tensor_from_array(inputs(6), fut_cat,    L3, torch_kCPU)

        !===============
        ! Wrap the output buffer
        call torch_tensor_from_array(outputs(1), out_quantiles, L3, torch_kCPU)
    
        !===============
        ! run pytorch model
        call torch_model_forward(model_pytorch, inputs, outputs)
        !===============
        ! extract median quantile (0.5)
        out_data_tft(1,:) = real(out_quantiles(1,:,2), kind=r8)
        !===============
        ! free pytorch model and tensors
        ! call torch_model_free(model_pytorch)                                       ! TO DO: confirm torch_model_free in ftorch
        do i = 1, n_in
          call torch_delete(inputs(i))
        end do
        call torch_delete(outputs(1))
        deallocate(inputs, outputs, L2, L3)

    end subroutine run_tft_model
  
end program MLPhenology
