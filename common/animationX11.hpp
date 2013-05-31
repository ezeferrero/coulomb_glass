#include <X11/Xlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include <time.h>
#include <blitz/array.h>
#include <blitz/array/stencil-et.h>
#include <blitz/funcs.h>

#include<iostream>
#include<fstream>
#include <string>

#include<cmath>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<sys/wait.h>
#include<fftw3.h>

using namespace blitz;


/* initializes state[N] with a seed */
extern void init_genrand(unsigned long s);
/* generates a random number on (0,1)-real-interval */
extern double genrand_real3(void);

#define   ANGLES (int) (360*64)

class peliX11
{
	private:
	XArc    *Dum1;
	XArc    *Dum2;
	unsigned short r1, r2;  
	
	public:
	int Dimx,Dimy;
	Display *d;
	Window w;
	char *msg;
	int s;
	XEvent e;
	unsigned int N1, N2;
	bool first;
	std::vector<XSegment> Seg;

	peliX11()
	{		                        
		/* open connection with the server */
		msg= new char[50];
		sprintf(msg ,"Hello, World!");
		
		d = XOpenDisplay(NULL);
		if (d == NULL) {
			fprintf(stderr, "Cannot open display\n");
			exit(1);
		}
 
		s = DefaultScreen(d);

		Dimx=Dimy=500;
                /* create window */
		w = XCreateSimpleWindow(d, RootWindow(d, s), 10, 10, Dimx, Dimy, 1,BlackPixel(d, s), WhitePixel(d, s));


		/* select kind of events we are interested in */
		XSelectInput(d, w, ExposureMask | KeyPressMask);

                /* map (show) the window */
		XMapWindow(d, w);
		//XMapRaised (d, w);
		cout << "X11 window creada\n"; //sleep(10);

		first=0;

		r1 = 2;
		r2 = 5;			
	};

	void set_nro_objects(int N){
		N1=N;
		//cout << N1 << " vortices, y " << N2 << " pins\n";
		if ((Dum1 = (XArc*) malloc(N1 * sizeof(XArc))) == NULL)
		{ cout << "init_setup: malloc failed for Dum1\n"; exit(1);}   		

		for (int i=0; i<N1; i++) {
			Dum1[i].width = r1; Dum1[i].height = r1;
			Dum1[i].angle1 = 0; Dum1[i].angle2 =  ANGLES;
		}
	};
	
	void add_circles(double Lx, double Ly, Array<double,1> &X)
	{
		double scalex=(double)Dimx/Lx;
		double scaley=(double)Dimy/Ly;
		double x,y;
		for (int p=0; p<N1; p++) 
		{
			x=X(p)-floor(X(p)/Lx)*Lx;
			//if(X(nz,p)> Lx || X(nz,p)<0) cout << x << " vs " << X(nz,p) << endl;
			y=p;
			Dum1[p].x = (short int)(x*scalex-r1*0.5);
			Dum1[p].y = Dimy - (short int)(y*scaley+r1*0.5);
		}
		//cout << "N1= " << N1 << endl;
	};

	void add_segments(double Lx, double Ly, double x1, double y1, double x2, double y2)
	{	
		double scalex=(double)Dimx/Lx;
		double scaley=(double)Dimy/Ly;
		XSegment seg;

		seg.x1=( short)(x1*scalex);
		seg.x2= (short)(x2*scalex);

		seg.y1= Dimy-(short)(y1*scaley);
		seg.y2= Dimy-(short)(y2*scaley);
	
		Seg.push_back(seg);
	};
	
	
	~peliX11(){
		XCloseDisplay(d);
	};
	
	void ver(){
		/* map (show) the window */
		cout << "watching\n";
		//XSync(d, 0);
		while (1) 
		{
			XNextEvent(d, &e);
			cout << e.type << " vs " << Expose << " vs " <<  KeyPress << endl;	
			if (e.type == Expose) 
			{
				XFillRectangle(d, w, DefaultGC(d, s), 20, 20, 10, 10);
				XDrawString(d, w, DefaultGC(d, s), 50, 50, msg, strlen(msg));
			}
			if (e.type == KeyPress)
			break;
			
		}			
	};

	void frame(){
		XClearWindow(d, w);

		XFillArcs(d, w, DefaultGC(d, s), Dum1, N1);

		int nsegments=Seg.size();
		if(nsegments>0)
 		XDrawSegments(d,w, DefaultGC(d, s), &Seg[0], nsegments);
		nsegments=0;
		
		XSync(d, 0);	
	};	
};

