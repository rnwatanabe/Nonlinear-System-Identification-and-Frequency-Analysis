module frolsfunctions
    use lapack95
    use f95_precision          
    implicit none
    include 'mkl.fi'       

    contains 

        recursive function fibonacci(term) result(fibo)
          integer, intent(in) :: term
          integer :: fibo
        
          if (term <= 1) then
            fibo = 1
          else
            fibo = fibonacci(term-1) + fibonacci(term-2)
          end if
          
        end function fibonacci
        
        function fiboseries(N) result(series)
            integer, intent(in) :: N
            integer :: series(N)
            integer :: i
            
            do i = 1, N
                series(i) = fibonacci(i)
            end do
        end function fiboseries
        
        recursive subroutine mfrols(p, y, pho, s, ESR, l, errorRR, A, q, g, verbose, sizeP, M, K, beta, M0)
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
            integer, parameter :: wp = kind(1.0d0)
            integer, intent(in) :: M, sizeP, K
            real(wp), intent(in) :: p(:,:,:)
            real(wp), intent(inout) :: A(M,M,K), q(sizeP,M,K)
            real(wp), intent(in) :: y(:,:)
            real(wp), intent(in) :: pho
            integer, intent(inout) :: s
            real(wp), intent(inout) :: ESR
            integer, intent(inout) :: l(M)
            real(wp), intent(inout) :: g(K,M)
            logical, intent(in), optional :: verbose
            logical :: verboseCond            
            real(wp), dimension(:,:), allocatable :: gs, errorRRTerm
            real(wp), intent(inout) :: errorRR(M)
            real(wp), dimension(:), allocatable :: ERR_m
            real(wp), allocatable :: qs(:,:,:)
            real(wp), allocatable :: sigma(:)
            integer :: term, r, i, j, nrhs, info
            integer :: rArray(s)
            real(wp), intent(out) :: beta(M, K)
            integer, intent(out) :: M0
            real(wp), dimension(:,:), allocatable :: Ainv
            real(wp), dimension(:,:), allocatable :: b   
            integer, dimension(:), allocatable :: ipiv
                   
            
            rArray = [(i, i=1,s, 1)]
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
            
            qs = p
            
            sigma = sum(y**2, dim=1)
            
            do term = 1, M
                if (all(l .ne. term)) then
!                     ## The Gram-Schmidt method was implemented in a modified way,
!                     ## as shown in Rice, JR(1966)                
                    
                    do r = 1, s-1
                        qs(:, term, :) = qs(:, term, :) - &
                                        spread(sum(q(:, r, :)*qs(:, term, :), dim=1) / & 
                                        sum(q(:, r, :)*q(:, r, :), dim=1), 1, sizeP)*q(:, r, :)
                    end do
                    
                    gs(:, term) = sum(y*reshape(qs(:, term, :), (/sizeP, K/)), dim=1)/&
                                  (sum(qs(:, term, :)*qs(:, term, :), dim=1) + 1e-6)
                    
                    errorRRTerm(:, term) = (gs(:, term)**2*sum(qs(:, term, :)*qs(:, term, :), dim=1)/sigma)
                end if
            end do        
        
        
            ERR_m = sum(errorRRTerm, dim=1)/K
            
            l(s) = maxloc(ERR_m, dim=1)
            errorRR(s) = ERR_m(l(s))
                       
            A(rArray, s, :) = sum(q(:, rArray, :)*spread(p(:, l(s), :), 2, size(rArray)), dim=1)/&
                              (sum(q(:, rArray, :)*q(:, rArray, :), dim=1))
            A(s, s, :) = 1.0
            q(:, s, :) = qs(:, l(s), :)
            g(:, s) = gs(:, l(s))
        
            ESR = ESR - errorRR(s)   
        
!             ## recursive call
        
            if (errorRR(s)>=pho .and. s<M) then
                if (verboseCond) then
                    print *, 'term number', s
                    print *, 'ERR', errorRR(s)
                end if
                s = s + 1
                deallocate(qs)
                deallocate(gs)
                call mfrols(p, y, pho, s, ESR, l, errorRR, A, q, g, verbose, sizeP, M, K, beta, M0)
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
            integer, parameter :: wp = kind(1.0d0)
            integer, intent(in) :: M, sizeP, K
            real(wp), intent(in) :: p(:,:,:)
            real(wp), intent(in) :: y(:,:)
            real(wp) :: A(M,M,K), q(sizeP,M,K), qs(sizeP,M,K)
            real(wp) :: g(K,M), gs(K, M)
            real(wp), dimension(M,K), intent(out) :: beta
            real(wp), dimension(M,1), intent(out) :: beta_m
            integer :: term, r, j, nrhs, info, M0
            real(wp) :: Ainv(M, M)
            real(wp) :: b(M,1)
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
            integer, parameter :: wp = kind(1.0d0)
            integer, intent(in) :: N
            real(wp) :: I(N,N)
            integer :: k
            
            I(:,:) = 0.0
            
            
            do k = 1, N
                I(k,k) = 1.0
            end do          
             
        end function
        
        subroutine RLS(p, y, lamb, Nmax, supress, sizeP, M, K, beta_m, beta)
            integer, parameter :: wp = kind(1.0d0)
            integer, intent(in) :: M, sizeP, K
            real(wp), intent(in) :: p(:,:,:)
            real(wp), intent(in) :: y(:,:)
            real(wp), intent(in) :: lamb
            integer, intent(in) :: Nmax
            logical, intent(in), optional :: supress
            logical :: supressCond
            real(wp) :: invLambda
            real(wp), dimension(M, K), intent(out) :: beta
            real(wp), dimension(M, 1), intent(out) :: beta_m
            real(wp) :: Pm(M,M,K)
            integer :: i, N, j
            real(wp), dimension(M, K) :: betaant
            real(wp), dimension(M, Nmax) :: e_beta
            real(wp) :: identMatrix(M,M)
            
            supressCond = .false.
            if (present(supress)) then
                supressCond = supress
            end if            
            
            invLambda = 1.0/lamb
            
            identMatrix = eye(M)
            
            Pm = spread(1e6*identMatrix, 3, K)         
            
            betaant = beta
                       
            i = 1
            
            do N = 1, Nmax
                do j = 1, K
                    Pm(:,:,j) = invLambda*(Pm(:,:,j) - &
                                (invLambda*matmul(matmul(matmul(Pm(:,:,j), reshape(p(i,:,j), (/M,1/))), &
                                reshape(p(i,:,j), (/1, M/))), Pm(:,:,j)))/ &
                                (1 + invLambda*dot_product(matmul(p(i,:,j), Pm(:,:,j)), p(i,:,j))))
                    beta(:,j) = beta(:,j) + matmul(Pm(:,:,j), p(i,:,j))*(y(i,j) - dot_product(p(i,:,j), beta(:,j)))
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
