#Double pendulum
from numpy import *
import math
import matplotlib.pyplot as plt
from matplotlib import animation

m=1.0 #m=m2/m1
l=1.0 #l=l2/l1

fig = plt.figure()
#ax = plt.axes(xlim=(-2.0, 2.0), ylim=(-2.0, 2.0),aspect='equal')
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-2, 2), ylim=(-2, 2))
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)                     
line, = ax.plot([], [], lw=2) 
line2, = ax.plot([], [],lw=2)
line3,=ax.plot([],[],lw=1)
point,=ax.plot([],[],'bo-')
time_template = 'time = %.1fs'
energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
#x1=theta_1, x2=theta_2, p1=theta1_dot, p2=theta2_dot
def init():
    time_text.set_text('')
    line.set_data([], [])
    line2.set_data([], [])
    line3.set_data([],[])
    energy_text.set_text('')
    point.set_data([],[])
    return line,  line2, line3, point, energy_text
    
def x1_dot(x1,x2,p1,p2):
	return p1
	
def x2_dot(x1,x2,p1,p2):
	return p2
	
def p1_dot(x1,x2,p1,p2):
        n=(2+m)*sin(x1)+m*sin(x1-2*x2)+2*m*sin(x1-x2)*(l*p2**2+cos(x1-x2)*p1**2)
	n=n*(-1.0)
	d=(2+m)-m*cos(2*x1-2*x2)
	return n/d
	
def p2_dot(x1,x2,p1,p2):
	n=2*sin(x1-x2)*(p1**2*(1+m)+(1+m)*cos(x1)+l*p2**2*m*cos(x1-x2))
	d=((2+m)-m*cos(2*x1-2*x2))*l
	return n/d

def k1(x1,x2,p1,p2,chse):
	if(chse==1):
		k1=x1_dot(x1,x2,p1,p2)
	if(chse==2):
		k1=x2_dot(x1,x2,p1,p2)
	if(chse==3):
		k1=p1_dot(x1,x2,p1,p2)
	if(chse==4):
		k1=p2_dot(x1,x2,p1,p2)
	return k1

def k2(x1,x2,p1,p2,chse,dt,k):
	x1=x1+k*dt/(2.0)
	x2=x2+k*dt/(2.0)
	p1=p1+k*dt/(2.0)
	p2=p2+k*dt/(2.0)
	if(chse==1):		
		k2=x1_dot(x1,x2,p1,p2)
	if(chse==2):
		k2=x2_dot(x1,x2,p1,p2)
	if(chse==3):
		k2=p1_dot(x1,x2,p1,p2)
	if(chse==4):
		k2=p2_dot(x1,x2,p1,p2)
	return k2


def k3(x1,x2,p1,p2,chse,dt,k):
	x1=x1+k*dt/(2.0)
	x2=x2+k*dt/(2.0)
	p1=p1+k*dt/(2.0)
	p2=p2+k*dt/(2.0)
	if(chse==1):		
		k3=x1_dot(x1,x2,p1,p2)
	if(chse==2):
		k3=x2_dot(x1,x2,p1,p2)
	if(chse==3):
		k3=p1_dot(x1,x2,p1,p2)
	if(chse==4):
		k3=p2_dot(x1,x2,p1,p2)
	return k3

def k4(x1,x2,p1,p2,chse,dt,k):
	x1=x1+k*dt
	x2=x2+k*dt
	p1=p1+k*dt
	p2=p2+k*dt
	if(chse==1):		
		k4=x1_dot(x1,x2,p1,p2)
	if(chse==2):
		k4=x2_dot(x1,x2,p1,p2)
	if(chse==3):
		k4=p1_dot(x1,x2,p1,p2)
	if(chse==4):
		k4=p2_dot(x1,x2,p1,p2)
	return k4

initial=zeros(4)

def runge_kutta(ti,tf,initial,dt):
	n=int((tf-ti)/dt)+1
	t=linspace(ti,tf,n)
	x1=zeros(n)
	x2=zeros(n)
	p1=zeros(n)
	p2=zeros(n)
	x1[0]=initial[0]
	x2[0]=initial[1]
	p1[0]=initial[2]
	p2[0]=initial[3]
	for i in range(n-1):
		k_1=k1(x1[i],x2[i],p1[i],p2[i],1)
		k_2=k2(x1[i],x2[i],p1[i],p2[i],1,dt,k_1)
		k_3=k3(x1[i],x2[i],p1[i],p2[i],1,dt,k_2)
		k_4=k4(x1[i],x2[i],p1[i],p2[i],1,dt,k_3)
		x1[i+1]=x1[i]+(k_1+2*k_2+2*k_3+k_4)*dt/6.0
		k_1=k1(x1[i],x2[i],p1[i],p2[i],2)
		k_2=k2(x1[i],x2[i],p1[i],p2[i],2,dt,k_1)
		k_3=k3(x1[i],x2[i],p1[i],p2[i],2,dt,k_2)
		k_4=k4(x1[i],x2[i],p1[i],p2[i],2,dt,k_3)
		x2[i+1]=x2[i]+(k_1+2*k_2+2*k_3+k_4)*dt/6.0
		k_1=k1(x1[i],x2[i],p1[i],p2[i],3)
		k_2=k2(x1[i],x2[i],p1[i],p2[i],3,dt,k_1)
		k_3=k3(x1[i],x2[i],p1[i],p2[i],3,dt,k_2)
		k_4=k4(x1[i],x2[i],p1[i],p2[i],3,dt,k_3)
		p1[i+1]=p1[i]+(k_1+2*k_2+2*k_3+k_4)*dt/6.0
		k_1=k1(x1[i],x2[i],p1[i],p2[i],4)
		k_2=k2(x1[i],x2[i],p1[i],p2[i],4,dt,k_1)
		k_3=k3(x1[i],x2[i],p1[i],p2[i],4,dt,k_2)
		k_4=k4(x1[i],x2[i],p1[i],p2[i],4,dt,k_3)
		p2[i+1]=p2[i]+(k_1+2*k_2+2*k_3+k_4)*dt/6.0
	return t,x1,x2,p1,p2,n

v0=zeros(4)
v0[0]=1.57
v0[1]=-0.33
v0[2]=0.0
v0[3]=0.0	
dt=0.01	
t,x1,x2,p1,p2,n=runge_kutta(0.0,50.0,v0,dt)
x_1=zeros(100)
y_1=zeros(100)
x_2=zeros(100)
y_2=zeros(100)
x=zeros(n)
y=zeros(n)

def energy(th1,th2,om1,om2):
    K=0.5*(1+m)*om1**2+0.5*m*l**2*om2**2+m*l*om1*om2*cos(th1-th2)
    V=-(1+m)*cos(th1)-m*l*cos(th2)
    return K+V
    
def test_ani(i):
    time_text.set_text(time_template%(i*dt))
    energy_text.set_text('energy = %.3f' %(energy(x1[i],x2[i],p1[i],p2[i])))
    y_1=linspace(-cos(x1[i]),0.0,100)
    x_1=-tan(x1[i])*y_1      
    y0=linspace(-l*cos(x2[i]),0.0,100)
    y_2=-cos(x1[i])+y0
    x_2=sin(x1[i])-tan(x2[i])*y0
    line.set_data(x_1,y_1)
    line2.set_data(x_2,y_2)
    
    xx=array([0,x_1[0],x_2[0]])
    yy=array([0,y_1[0],y_2[0]])
    point.set_data(xx,yy)
    x=zeros(i)
    y=zeros(i)
   
    for k in range(i):
        y_1=linspace(-cos(x1[k]),0.0,100)
        x_1=-tan(x1[k])*y_1 
        y0=linspace(-l*cos(x2[k]),0.0,100)
        y_2=-cos(x1[k])+y0
        x_2=sin(x1[k])-tan(x2[k])*y0 
        x[k]=x_2[0]
        y[k]=y_2[0]
          
    line3.set_data(x,y)
   
    return line, line2, line3,time_text, energy_text,point

'''        
def test_ani2(i):
    x=[x_2[0]]
    y=[y_2[0]]
    line3.set_data(x,y)
    return line3
'''    
for i in range(n):
    y=-cos(x1[i])
    x=-tan(x1[i])*y 
    plt.plot(x,y,lw=2)
           
from time import time
t0 = time()
test_ani(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)
print interval
ax.grid()    
anim = animation.FuncAnimation(fig, test_ani, frames=n,init_func=init, interval=interval,blit=True)  
#anim.save('double_pendulum_trial.mp4', fps=30, extra_args=['-vcodec', 'libx264'])                        
plt.show()						
	
