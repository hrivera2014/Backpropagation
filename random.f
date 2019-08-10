        implicit none
        integer i
        integer itera,ipuntos
        parameter(itera=10,ipuntos=30)
        real x,y(itera,ipuntos)
        real*8 xr
        integer iran,icont
        iran=18354
        iran=68354

        open (unit=18,file='random.dat') 
        do i=1,itera
        icont=0 
        do x=1,ipuntos
        icont=icont+1
        call rnd001(xr,iran,1)
        y(i,icont)=xr
        enddo
        enddo

        
        icont=0 
        do x=1,ipuntos
        icont=icont+1
        write(18,100) y(1,x),y(2,x),y(3,x),y(4,x),y(5,x),y(6,x),
     2  y(7,x),y(8,x),y(9,x),y(10,x)
 100    format(10(1x,f6.3))
        enddo

        
        stop
        end

c Función random

        subroutine rnd001(xi,i,ifin)
        integer*4 i,ifin
        real*8 xi
        i=i*54891
        xi=i*2.328306e-10+0.5D00
        xi=xi*ifin
        return
        end
