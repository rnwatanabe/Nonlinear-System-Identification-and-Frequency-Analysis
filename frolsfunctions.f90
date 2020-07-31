module frolsfunctions
    use lapack95
    use blas95
    use f95_precision          
    implicit none
    ! gfortran -c ~/intel/mkl2019/mkl/include/lapack.f90
    ! gfortran -c ~/intel/mkl2019/mkl/include/blas.f90
    ! gfortran -c -I/home/rnwatanabe/intel/mkl2019/compilers_and_libraries_2019.5.281/linux/mkl/include/ ~/Dropbox/Nonlinear-System-Identification-and-Frequency-Analysis/frolsfunctions.f90
    ! ~/miniconda3/bin/python -m numpy.f2py -I/home/rnwatanabe/intel/mkl2019/mkl/include/ -L/home/rnwatanabe/intel/mkl2019/mkl/lib/intel64/libmkl_gf_ilp64.a -L/home/rnwatanabe/intel/mkl2019/mkl/lib/intel64/libmkl_gnu_thread.a -L/home/rnwatanabe/intel/mkl2019/mkl/lib/intel64/libmkl_core.a -lgomp -lpthread -lm -ldl -c frolsfunctions.f90 -m frolsfunctions

    contains 

        recursive subroutine mfrols(p, y, pho, s, ESR, l, errorRR, A, qs, g, verbose, sizeP, M, K, beta, M0)
    !         '''
    !         Implements the MFROLS algorithm (see page 97 from Billings, SA (2013)).
    !             written by: Renato Naville Watanabe
    !             beta = mfrols(p, y, pho, s)
    !             Inputs:
    !               p: matrix of floats, is the matrix of candidate terms.
    !               y: vector of floats, output signal.
    !               pho: float, stop criteria.
    !               s: integer, iteration step of the mfrols algorithm.
    !               l: vector of integers, indices of the chosen terms.M = np.shape(p)[1]; l = -1*np.ones((M))
    !               err: vector of floats, the error reduction ratio of each chosen term. err = np.zeros((M))
    !               ESR: float, the sum of the individual error reduction ratios. Initial value eual 1.
    !               A: matrix of floats, auxiliary matrix in the orthogonalization process.
    !                       A = np.zeros((M,M,1))
    !               q: matrix of floats, matrix with each column being the terms orthogonalized
    !                       by the Gram-Schmidt process. q = np.zeros_like(p)
    !               g: vector of floats, auxiliary vector in the orthogonalization process.
    !                       g = np.zeros((1,M))
    !             Output:
    !               beta: vector of floats, coefficients of the chosen terms.
    !               l: vector of integers, indices of the chosen terms
    !               M0: number of chosen terms
    !         '''
            implicit none
            include 'mkl.fi'
            integer, parameter :: wp = kind(1.0d0)
            integer, intent(in) :: M, sizeP, K
            real(kind=wp), intent(inout) :: p(:,:,:)
            real(kind=wp), intent(inout) :: A(M,M,K), qs(sizeP,M,K)
            real(kind=wp), intent(in) :: y(:,:)
            real(kind=wp), intent(in) :: pho
            integer, intent(inout) :: s
            real(kind=wp), intent(inout) :: ESR
            integer, intent(inout) :: l(M)
            real(kind=wp), intent(inout) :: g(K,M)
            logical, intent(in), optional :: verbose
            logical :: verboseCond            
            real(kind=wp), dimension(:,:), allocatable :: gs, errorRRTerm
            real(kind=wp), intent(inout) :: errorRR(M)
            real(kind=wp), dimension(:), allocatable :: ERR_m
            real(kind=wp), allocatable :: sigma(:)
            integer :: term, j, nrhs, info
            real(kind=wp), intent(out) :: beta(M, K)
            integer, intent(out) :: M0
            real(kind=wp), dimension(:,:), allocatable :: Ainv
            real(kind=wp), dimension(:,:), allocatable :: b   
            integer, dimension(:), allocatable :: ipiv           
            
            verboseCond = .false.
            if (present(verbose)) then
                verboseCond = verbose
            end if
            
            allocate(gs(K, M))
            gs = 0.0
            allocate(ERR_m(K))
            ERR_m = 0.0
            allocate(errorRRTerm(K, M))
            errorRRTerm = 0.0
            allocate(sigma(K))
            
            sigma = sum(y**2, dim=1)
            
            do term = 1, M
                if (all(l .ne. term)) then
!                     ## The Gram-Schmidt method was implemented in a modified way,
!                     ## as shown in Rice, JR(1966)                
                    
                    if (s>1) then
                        qs(:, term, :) = qs(:, term, :) - &
                                        spread(sum(qs(:, l(s-1), :)*qs(:, term, :), dim=1) / & 
                                        sum(qs(:, l(s-1), :)*qs(:, l(s-1), :), dim=1), 1, sizeP)*qs(:, l(s-1), :)
                    end if
                    
                    gs(:, term) = sum(y*reshape(qs(:, term, :), (/sizeP, K/)), dim=1)/&
                                  (sum(qs(:, term, :)*qs(:, term, :), dim=1) + 1e-6)
                    
                    errorRRTerm(:, term) = (gs(:, term)**2*sum(qs(:, term, :)*qs(:, term, :), dim=1)/sigma)
                end if
            end do        
        
        
            ERR_m = sum(errorRRTerm, dim=1)/K
            
            l(s) = maxloc(ERR_m, dim=1)
            errorRR(s) = ERR_m(l(s))
                       
            A(1:s-1, s, :) = sum(qs(:, l(1:s-1), :)*spread(p(:, l(s), :), 2, s-1), dim=1)/&
                              (sum(qs(:, l(1:s-1), :)*qs(:, l(1:s-1), :), dim=1))
            A(s, s, :) = 1.0
            
            g(:, s) = gs(:, l(s))
        
            ESR = ESR - errorRR(s)   
        
!             ## recursive call
        
            if (errorRR(s)>=pho .and. s<M) then
                if (verboseCond) then
                    print *, 'term number', s
                    print *, 'ERR', errorRR(s)
                end if
                s = s + 1
                deallocate(gs)
                call mfrols(p, y, pho, s, ESR, l, errorRR, A, qs, g, verbose, sizeP, M, K, beta, M0)
            else
                if (verboseCond) then
                    print *, 'term number', s
                    print *, 'ERR', errorRR(s)
                end if
                M0 = s
                s = s + 1
                
                allocate(Ainv(M0,M0))
                allocate(b(M0,1))
                allocate(ipiv(M0))
                nrhs = 1
                do j = 1, K
                    if (s > 1) then
                        Ainv = A(1:M0, 1:M0, j)
                        b = reshape(g(j, 1:M0), (/M0, nrhs/))
                        call dgesv(M0, nrhs, Ainv, M0, ipiv, b, M0, info)
                        beta(1:M0, j) = reshape(b, (/M0/))
                    else
                        beta(1, j) = (A(1,1,j)**(-1))*g(j, 1)
                    end if
                end do
            end if
        end subroutine
        
        subroutine mols(p, y, sizeP, M, K, beta_m, beta)
            implicit none
            include 'mkl.fi'
            integer, parameter :: wp = kind(1.0d0)
            integer, intent(in) :: M, sizeP, K
            real(kind=wp), intent(in) :: p(:,:,:)
            real(kind=wp), intent(in) :: y(:,:)
            real(kind=wp) :: A(M,M,K), q(sizeP,M,K), qs(sizeP,M,K)
            real(kind=wp) :: g(K,M), gs(K, M)
            real(kind=wp), dimension(M,K), intent(out) :: beta
            real(kind=wp), dimension(M,1), intent(out) :: beta_m
            integer :: term, r, j, nrhs, info, M0
            real(kind=wp) :: Ainv(M, M)
            real(kind=wp) :: b(M,1)
            integer :: ipiv(M)  
        
            qs = p
            M0 = M
            A = 0.0
            q = 0.0
            g = 0.0
            do term = 1, M
!                 ## The Gram-Schmidt method was implemented in a modified way, as shown in Rice, JR(1966)
                do r = 1, term-1
                    qs(:, term, :) = qs(:, term, :) - spread(sum(q(:, r, :)*qs(:, term, :), dim=1)/ &
                                                    (sum(q(:, r, :)*q(:, r, :), dim=1)+1e-6), 1, sizeP)*q(:, r, :)
                    A(r, term, :) = sum(q(:, r, :)*p(:, term, :), dim=1)/(sum(q(:, r, :)*q(:, r, :), dim=1)+1e-6) 
                end do
                gs(:, term) = sum(y*qs(:, term, :), dim=1)/(sum(qs(:, term, :)*qs(:, term, :), dim=1)+1e-6)
                A(term, term, :) = 1.0
                q(:, term, :) = qs(:, term, :)
                g(:, term) = gs(:, term)
            end do
            nrhs = 1
            do j = 1, K
                if (M > 1) then
                    Ainv = A(:, :, j)
                    b = reshape(g(j, :), (/M0, nrhs/))
                    call dgesv(M0, nrhs, Ainv, M0, ipiv, b, M0, info)
                    beta(:, j) = reshape(b, (/M/))
                else
                    beta(1, j) = (A(1,1,j)**(-1))*g(j, 1)
                end if
            end do
            beta_m = reshape(sum(beta, dim=2)/K, (/M,1/))
        end subroutine
        
        function eye(N) result(I)
            implicit none
            integer, parameter :: wp = kind(1.0d0)
            integer, intent(in) :: N
            real(kind=wp) :: I(N,N)
            integer :: k
            
            I(:,:) = 0.0
            
            
            do k = 1, N
                I(k,k) = 1.0
            end do          
             
        end function
        
        subroutine RLS(p, y, lamb, Nmax, sizeP, M, K, supress, beta_m, beta)
            implicit none
            include 'mkl.fi'
            integer, parameter :: wp = kind(1.0d0)
            real(kind=wp), intent(in) :: p(sizeP,M,K)
            real(kind=wp), intent(in) :: y(sizeP,K)
            real(kind=wp), intent(in) :: lamb
            integer, intent(in) :: Nmax
            integer, intent(in) :: M, sizeP, K
            logical, intent(in), optional :: supress
            logical :: supressCond
            real(kind=wp) :: invLambda
            real(kind=wp), dimension(M, K), intent(out) :: beta
            real(kind=wp), dimension(M, 1), intent(out) :: beta_m
            real(kind=wp) :: Pm(M,M,K)
            integer :: i, N, j
            real(kind=wp), dimension(M, K) :: betaant
            real(kind=wp), dimension(M, Nmax) :: e_beta
            real(kind=wp) :: identMatrix(M,M)
            character*1 :: transa, transb
            real(kind=wp) :: alpha, gamma
            integer :: ia
            real(kind=wp) :: pmp(M), pmpp(M,M), pmpppm(M,M), ppm(M)
            real(wp) :: pbeta, ppmp
            real(kind=wp) :: A(M,M), b(M), C(M,1), D(1, M)
        
            
            supressCond = .false.
            if (present(supress)) then
                supressCond = supress
            end if            
            
            invLambda = 1.0/lamb
            
            identMatrix = eye(M)
            
            Pm = spread(1e6*identMatrix, 3, K)         
            
            betaant = beta
                       
            i = 1
            transa = 'N'
            transb = 'T'
            alpha = 1.0
            gamma = 0.0
            ia = 1
            
            do N = 1, Nmax
                do j = 1, K
                    A = Pm(:,:,j)
                    b = p(i,:,j)
!                    pmp = matmul(A, b)
                    call dgemv(transa, M, M, alpha, A, M, b, ia, gamma, pmp, ia)
!                    pmpp = matmul(reshape(pmp, (/M,1/)), reshape(p(i,:,j), (/1, M/)))                 
                    C = reshape(pmp, (/M,1/))
                    D = reshape(p(i,:,j), (/1, M/))
                    call dgemm(transa, transa, M, M, ia, alpha, &
                               C, M, D, &
                               ia, gamma, pmpp, M)
!                    pmpppm = matmul(pmpp, A)
                    call dgemm(transa, transa, M, M, M, alpha, &
                                pmpp, M, A, M, gamma, pmpppm, M)
!                    ppm = matmul(b, A)
                    call dgemv(transb, M, M, alpha, A, M, b, ia, gamma, ppm, ia)
!                    pbeta = dot_product(p(i,:,j), beta(:,j))
                    pbeta = ddot(M, p(i,:,j), ia, beta(:,j), ia)
!                    ppmp = dot_product(ppm, p(i,:,j))
                    ppmp = ddot(M, ppm, ia, p(i,:,j), ia)
                    Pm(:,:,j) = invLambda*(Pm(:,:,j) - (invLambda*pmpppm)/&
                                                       (1 + invLambda*ppmp))
                    A = Pm(:,:,j)
!                    pmp = matmul(A, b)
                    call dgemv(transa, M, M, alpha, A, M, b, ia, gamma, pmp, ia)
                    beta(:,j) = beta(:,j) + pmp*(y(i,j) - pbeta)
                end do
                e_beta(:, N) = sum((beta - betaant)**2, dim=2)/K
                betaant = beta
                i = i + 1
                if (i > sizeP) then
                    i = 1
                    if (.not. supress) then
                        print *, N, sum(e_beta(:,N-1))/M
                    end if
                end if
            end do
            
            beta_m = reshape(sum(beta, dim=2)/K, (/M, 1/))
            
        end subroutine   
end module frolsfunctions
