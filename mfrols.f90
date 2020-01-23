module frolsfunctions
    implicit none
    integer, parameter :: wp = kind(1.0d0)
    real(wp), parameter :: pi = 4 * atan(1.0_wp)    

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
        
end module frolsfunctions
