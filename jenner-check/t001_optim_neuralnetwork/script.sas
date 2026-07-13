/**********************************************************************
 * Notas de Modelos Predictivos;
 * Jose Enrique Perez ;
 * Maestría en Ciencias Actuariales;
 * Facultad de Negocios. Universidad La Salle México;
 * Purpose: To classify images with a neural network.
 **********************************************************************/

/* Hyperparameters */
%let iniVal = 0.25;
%let lambda1=0;
%let lambda2=0.1;

proc optmodel printlevel=2;

	ods output PrintTable=expt ProblemSummary=exps DerivMethods=exdm
	              SolverOptions=exso SolutionSummary=exss OptStatistics=exos
	              Timing=exti;
	
	/* Weights */
	var w110 init &iniVal., w111 init &iniVal., w120 init &iniVal., w121 init &iniVal., w130 init &iniVal., w131 init &iniVal., w140 init &iniVal., w141 init &iniVal.,
	w210 init &iniVal., w211 init &iniVal., w212 init &iniVal., w213 init &iniVal., w214 init &iniVal.,
	w220 init &iniVal., w221 init &iniVal., w222 init &iniVal., w223 init &iniVal., w224 init &iniVal.,
	w230 init &iniVal., w231 init &iniVal., w232 init &iniVal., w233 init &iniVal., w234 init &iniVal.,
	w0 init &iniVal., w1 init &iniVal., w2 init &iniVal., w3 init &iniVal.
	;
	
	/* 1st hidden layer */
	impvar H11_2 = 1/(1+exp(w110+w111*2));
	impvar H11_3 = 1/(1+exp(w110+w111*3));
	impvar H12_2 = 1/(1+exp(w120+w121*2));
	impvar H12_3 = 1/(1+exp(w120+w121*3));
	impvar H13_2 = 1/(1+exp(w130+w131*2));
	impvar H13_3 = 1/(1+exp(w130+w131*3));
	impvar H14_2 = 1/(1+exp(w140+w141*2));
	impvar H14_3 = 1/(1+exp(w140+w141*3));
	/* 2nd hidden layer */
	impvar H21_2 = 1/(1+exp(w210+w211*H11_2+w212*H12_2+w213*H13_2+w214*H14_2));
	impvar H22_2 = 1/(1+exp(w220+w221*H11_2+w222*H12_2+w223*H13_2+w224*H14_2));
	impvar H23_2 = 1/(1+exp(w230+w231*H11_2+w232*H12_2+w233*H13_2+w234*H14_2));
	impvar H21_3 = 1/(1+exp(w210+w211*H11_3+w212*H12_3+w213*H13_3+w214*H14_3));
	impvar H22_3 = 1/(1+exp(w220+w221*H11_3+w222*H12_3+w223*H13_3+w224*H14_3));
	impvar H23_3 = 1/(1+exp(w230+w231*H11_3+w232*H12_3+w233*H13_3+w234*H14_3));
	/* Regularization terms */
	impvar L1 = 
	abs(w110)+abs(w111)+abs(w120)+abs(w121)+abs(w130)+abs(w131)+abs(w140)+abs(w141)+
	abs(w210)+abs(w211)+abs(w212)+abs(w213)+abs(w214)+
	abs(w220)+abs(w221)+abs(w222)+abs(w223)+abs(w224)+
	abs(w230)+abs(w231)+abs(w232)+abs(w233)+abs(w234)+
	abs(w0)+abs(w1)+abs(w2)+abs(w3);
	impvar L2 = 
	(w110)**2+(w111)**2+(w120)**2+(w121)**2+(w130)**2+(w131)**2+(w140)**2+(w141)**2+
	(w210)**2+(w211)**2+(w212)**2+(w213)**2+(w214)**2+
	(w220)**2+(w221)**2+(w222)**2+(w223)**2+(w224)**2+
	(w230)**2+(w231)**2+(w232)**2+(w233)**2+(w234)**2+
	(w0)**2+(w1)**2+(w2)**2+(w3)**2;
	
	/* Loss function to minimize  */
	min f = ((1-w0-w1*H21_2-w2*H22_2-w3*H23_2)**2+(0-w0-w1*H21_3-w2*H22_3-w3*H23_3)**2)/2 + &lambda1.*L1+&lambda2./2*L2;
	
	/* lbfgs means Limited-Memory Broyden-Fletcher-Goldfarb-Shanno */ 
	solve with nlpu  / tech = lbfgs ;
	print w110 w111 w120 w121 w130 w131 w140 w141
	w210 w211 w212 w213 w214
	w220 w221 w222 w223 w224
	w230 w231 w232 w233 w234
	w0 w1 w2 w3;
 
quit;

