c Copyright 2019 Hender Rivera

c ******** Algoritmo de retropropagación de errores para el
c entrenamiento de redes neuronales artificiales multicapa. 
        implicit double precision (a-h,o-z)
c ia=tamano del vector de entrenamiento y ia-1 # de 
c neuronas de la primera capa, ib=# de neuronas de de la capa
c oculta, ic=# de neuronas de la capa de salida.        
        parameter(ia=31,ib=20,ic=2)
c Número de capas y coediciente de aprendisaje       
        parameter(inc=3,eta=-0.10)
        integer*4 iran
        real*8 xr
        real xm,sse
c        common /wei1/ pesos(ia-1,ia,1)
c        common /wei2/ pesos(ib,ia-1,2)
c        common /wei3/ pesos(ic,ib,3)
        common /wei1/ pesos1(ia-1,ia)
        common /wei2/ pesos2(ib,ia-1)
        common /wei3/ pesos3(ic,ib)
        common /aj1/ ajustew1(ia-1,ia)
        common /aj2/ ajustew2(ib,ia-1)
        common /aj3/ ajustew3(ic,ib)
        common /vec1/ vector1(31,10)
        common /vec2/ vector2(31,10)
        common /vec/ vector(31,10,2)
        common /etq1/ etiqueta1(2)
        common /etq2/ etiqueta2(2)
        common /etq/ etiqueta(2,2)
        common /sum1/ x1(ia)
        common /sum2/ x2(ib)
        common /sum3/ x3(ic)
        common /funact1/ y1(ia-1)
        common /funact2/ y2(ib+1)
        common /funact3/ y3(ic+1)
        common /grad1/ delta1(ia-1)
        common /grad2/ delta2(ib)
        common /grad3/ delta3(ic)
        real X1,X2,Y1,Y2
        real*4 cr,cg,cb
        real aspect,width
        integer PGOPEN
        integer PGQID
        integer ID
        logical graficos
c para esta semilla da        
        iran=98377
        graficos=.false.
        iunidad=16
        open(unit=9,file='random.dat')
        open(unit=10,file='patron1.dat')
        open(unit=11,file='patron2.dat')
        open(unit=12,file='etiqueta1.dat')
        open(unit=13,file='etiqueta2.dat')
        open(unit=14,file='pesos1.dat')
        open(unit=15,file='pesos2.dat')
        open(unit=16,file='pesos3.dat')
        open(unit=33,file='error_patron1.dat')
        open(unit=34,file='error_patron2.dat')
        
c  Sección de Gráficos
        if(graficos)then
c  Apertura de la ventana gráfica
        if (PGOPEN('/XSERVE') .le. 0) stop
c       if (PGOPEN('salida.ps/PS').LE.0)STOP
c  Borra todas las ventanas gráficas
        call PGERAS
        call PGENV(0.0,itamano*1.0+1,0.0,jtamano*1.0,0,-1)
c  Ajusta el aspect ratio AR=heigth/width
        aspect=6
        width=1
        call PGPAP(aspect,width)
        endif

c ****** Inicialización del los pesos
c        call wei(ia,ia,1)
c        call wei(ia,ib,2)
c        call wei(ib,ic,3)

c Inicialización aleatoria de los pesos entre el vector de 
c presentación y la primera capa
        do j=1,ia
        do i=1,ia-1
                call rnd001(xr,iran,1)
                sr=xr
                call rnd001(xr,iran,1)
                if(sr.lt.0.5)then
                        s=-1
                else
                        s=1
                        endif
                xr=xr*0.3*s
                pesos1(i,j)=xr
        enddo
        enddo
                
c Inicialización aleatoria de los pesos entre la capa de entrada 
c y la capa oculta
        do j=1,ia-1
        do i=1,ib
                call rnd001(xr,iran,1)
                sr=xr
                call rnd001(xr,iran,1)
                if(sr.lt.0.5)then
                        s=-1
                else
                        s=1
                        endif
                xr=xr*0.3*s
                pesos2(i,j)=xr
        enddo
        enddo

c Inicialización aleatoria de los pesos entre la capa oculta 
c y la capa de salida
        do j=1,ib
        do i=1,ic
                call rnd001(xr,iran,1)
                sr=xr
                call rnd001(xr,iran,1)
                if(sr.lt.0.5)then
                        s=-1
                else
                        s=1
                        endif
                xr=xr*0.3*s
                pesos3(i,j)=xr
        enddo
        enddo

c ****** 
c  Asignacion de los Bias 
        do j=1,10        
                vector1(1,j)=1
                vector2(1,j)=1
                vector(1,j,1)=1
                vector(1,j,2)=1
        enddo

c  Lectura de los patrones de entrenamiento
        
        do i=2,ia
        read(10,100) vector1(i,1),vector1(i,2),vector1(i,3),
     + vector1(i,4),vector1(i,5),vector1(i,6),vector1(i,7),
     + vector1(i,8),vector1(i,9),vector1(i,10)   
        read(11,100) vector2(i,1),vector2(i,2),vector2(i,3),
     + vector2(i,4),vector2(i,5),vector2(i,6),vector2(i,7),
     + vector2(i,8),vector2(i,9),vector2(i,10)   
        enddo
 
 100    format(10(1x,f6.3))
        
        do n=1,10
        do k=1,ia
        vector(k,n,1)=vector1(k,n)
        vector(k,n,2)=vector2(k,n)
        enddo
        enddo
       
c       Lectura de las etiquetas ó salida deseada

        read(12,*) etiqueta1(1),etiqueta1(2)
        read(13,*) etiqueta2(1),etiqueta2(2)
                etiqueta(1,1)=etiqueta1(1)
                etiqueta(1,2)=etiqueta1(2)
                etiqueta(2,1)=etiqueta2(1)
                etiqueta(2,2)=etiqueta2(2)

c ***********************************
c   Fase de entrenamiento de la red
c ***********************************

        epoch=0
        sse=1

 80     continue
        epoch=epoch+1

        if(epoch.gt.100000.or.sse.lt.0.005) goto 900
        
c        do 555 n=1,2
        n=2 
        do 400 l=1,10 
c        call rnd001(xr,iran,2)
c        n=xr

c ****** Fase de presentación del patrón
c Suma ponderada de las entradas a las neuronas de la  primera capa
c y la función de activación de esas neuronas
        y1(1)=1 
        do i=1,ia-1
        x1(i)=0
        y1(i+1)=0
        do j=1,ia 
        x1(i)=x1(i)+(pesos1(i,j)*vector(j,l,n))
c        x1(i)=x1(i)+(pesos1(i,j)*vector1(j,l))
        enddo
        y1(i+1)=1/(1+exp(-1*x1(i)))
        enddo

c Suma ponderada de las entradas a las neuronas de la capa oculta
c y la función de activación de esas neuronas
        y2(1)=1
        do i=1,ib
        x2(i)=0
        y2(i+1)=0
        do j=1,ia-1
        x2(i)=x2(i)+(pesos2(i,j)*y1(j))
        enddo
        y2(i+1)=1/(1+exp(-1*x2(i)))
        enddo

c Suma ponderada de las entradas a las neuronas de la capa de salida
c y la función de activación de esas neuronas
        do i=1,ic
        x3(i)=0
        y3(i)=0
        do j=1,ib
        x3(i)=x3(i)+(pesos3(i,j)*y2(j))
        enddo
        y3(i)=1/(1+exp(-1*x3(i)))
        enddo
c Cálculo del error cuadrático medio
        do i=1,ic
        sse=sse+(y3(i)-etiqueta(n,i))**2
c        sse=sse+(y3(i)-etiqueta1(i))**2
        enddo
        sse=0.5*sse

        if(mod(epoch,100).eq.0) print *,'sse=',sse,' epoch=',epoch

c        if(mod(epoch,100).eq.0) write(34,*) sse
c        write(33,*) sse

c ****** Fase de aprendisaje
c ** Paso atrás; propagando lo errores hacia atrás. 
        
c  1 Como influye el cambio de y3 en la capa de salida        
        do i=1,ic
        delta3(i)=y3(i)*(1-y3(i))*(y3(i)-etiqueta(n,i))
c        delta3(i)=y3(i)*(1-y3(i))*(y3(i)-etiqueta1(i))
        enddo 

c  2 Como influye el cambio de y2 en la capa oculta
        do j=1,ib+1
        sumdelta2=0
        do i=1,ic
        sumdelta2=sumdelta2+(delta3(i)*pesos3(i,j))
        enddo
        delta2(j)=y2(j)*(1-y2(j))*sumdelta2
        enddo

c  3 Como influye el cambio de y1 en la capa de entrada
        do j=1,ia-1
        sumdelta1=0 
        do i=1,ib
        sumdelta1=sumdelta1+(delta2(i)*pesos2(i,j))
        enddo
        delta1(j)=y1(j)*(1-y1(j))*sumdelta1
        enddo

c *** Cálculo del ajuste de los pesos 

c 1 Calcula el ajuste de los pesos entre la capa de salida y la capa oculta
        do i=1,ib+1
        do j=1,ic
        ajustew3(i,j)=eta*delta3(j)*y2(i)
        enddo
        enddo
        
c 2 Calcula el ajuste de los pesos entre la capa oculta y la capa de entrada 
        do i=1,ia-1
        do j=1,ib+1
        ajustew2(i,j)=eta*delta2(j)*y1(i)
        enddo
        enddo

c 3 Calcula el ajuste de los pesos entre la capa oculta y la capa de entrada 
        do j=1,ia-1
        do i=1,ia
        ajustew1(i,j)=eta*delta1(j)*vector(i,l,n)
c        ajustew1(i,j)=eta*delta1(j)*vector1(i,l)
        enddo
        enddo

c *** Ajuste de los pesos
c Se hacen los ajustes de los pesos 1
        do j=1,ia 
        do i=1,ia-1
        pesos1(i,j)=pesos1(i,j)+ajustew1(i,j) 
        enddo
        enddo
        
c Se hacen los ajustes de los pesos 2
        do j=1,ia-1
        do i=1,ib
        pesos2(i,j)=pesos2(i,j)+ajustew2(i,j) 
        enddo
        enddo

c Se hacen los ajustes de los pesos 3
        do j=1,ib
        do i=1,ic
        pesos3(i,j)=pesos3(i,j)+ajustew3(i,j) 
        enddo
        enddo

c---------------------------------------------------------------------72
 400    continue 

        write(34,*) sse
c 555    continue

        goto 80  

c *** Fin del programa ***
 900    continue

c Escribe los pesos

        write (14,140) pesos1
        write (15,150) pesos2
        write (16,160) pesos3
 140    format(t1,31(1x,f9.3))
 150    format(t1,20(1x,f9.3))
 160    format(t1,2(1x,f9.3))


c ***********************************************************
c Fase de reconocimiento de patrones una vez entrenada la red
c ***********************************************************
        n=2 
        do 444 l=1,10        
c Suma ponderada de las entradas a las neuronas de la  primera capa
c y la función de activación de esas neuronas
        y1(1)=1 
        do i=1,ia-1
        x1(i)=0
        y1(i+1)=0
        do j=1,ia 
        x1(i)=x1(i)+(pesos1(i,j)*vector(j,l,n))
c        x1(i)=x1(i)+(pesos1(i,j)*vector1(j,l))
        enddo
        y1(i+1)=1/(1+exp(-1*x1(i)))
        enddo

c Suma ponderada de las entradas a las neuronas de la capa oculta
c y la función de activación de esas neuronas
        y2(1)=1
        do i=1,ib
        x2(i)=0
        y2(i+1)=0
        do j=1,ia-1
        x2(i)=x2(i)+(pesos2(i,j)*y1(j))
        enddo
        y2(i+1)=1/(1+exp(-1*x2(i)))
        enddo

c Suma ponderada de las entradas a las neuronas de la capa de salida
c y la función de activación de esas neuronas
        do i=1,ic
        x3(i)=0
        y3(i)=0
        do j=1,ib
        x3(i)=x3(i)+(pesos3(i,j)*y2(j))
        enddo
        y3(i)=1/(1+exp(-1*x3(i)))
        enddo
        
        e1=(y3(1)-etiqueta(n,1))**2
c        e1=(y3(1)-etiqueta1(1))**2
        e2=(y3(2)-etiqueta(n,2))**2
c        e2=(y3(2)-etiqueta1(2))**2
        sse=.5*(e1+e2) 
        
        
        if (sse.lt.0.02) then 
          write(6,999) l,' patron',n,' y31=',y3(1),' y32=',y3(2),
     2      ' etiquetas',etiqueta(n,1),etiqueta(n,2),' sse=',sse
        else
          write(6,998) l,' irreconocible',' y31=',y3(1),' y32=',y3(2),
     2      ' etiquetas',etiqueta(n,1),etiqueta(n,2),' sse=',sse
        endif
        
 999    format(t1,i4,a7,i4,a5,f5.3,a5,f5.3,a10,2(f5.1),a5,f5.3) 
 998    format(t1,i4,a14,a5,f5.3,a5,f5.3,a10,2(f5.1),a5,f5.3) 
 444    continue        
        stop
        end
c---------------------------------------------------------------------72
c                      Funciones y subrutinas
c---------------------------------------------------------------------72
c Función random

        subroutine rnd001(xi,i,ifin)
        integer*4 i,ifin
        real*8 xi
        i=i*54891
        xi=i*2.328306e-10+0.5D00
        xi=xi*ifin
        return
        end

