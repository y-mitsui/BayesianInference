#include<gsl/gsl_rng.h>
#include<gsl/gsl_randist.h>
#include<gsl/gsl_statistics.h>
#include <gsl/gsl_integration.h>

#define EPSABS 1e-4
#define EPSREL 1e-4
double f2(double x,void *params){
	//double y=*((double*)params);
	double alpha[]={30,50};
	double theta[]={x,1.0-x};
	/*double theta2[]={0.3753639,0.6246361};
	double theta3[]={0.2,0.78};*/
	double r1=gsl_ran_dirichlet_pdf(2,alpha,theta);
	printf("%f\t%f\t%f\n",theta[0],theta[1],r1);
	/*double r2=gsl_ran_dirichlet_pdf(2,alpha,theta3);
	printf("r2:%f\n",r2);*/
	return r1;
}
double f1(double x,void *params){
	double result,error;
	size_t neval;
	gsl_function F;
	F.function = &f2;
	F.params =&x;
	int code=gsl_integration_qng(&F,0.0,1.0,EPSABS,EPSREL,&result,
                                &error,&neval);
	if(code!=0){
		fprintf(stderr,"error\n");
	}
	return result;
}
int main(void){
	double result,error;
	size_t neval;
	gsl_function F;
	F.function = &f2;
	int code=gsl_integration_qng(&F,0.0,1.0,EPSABS,EPSREL,&result,
                                &error,&neval);
	if(code!=0){
		fprintf(stderr,"error\n");
	}
	
	printf("r:%f\n",result);
	return 0;
}
