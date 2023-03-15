/**********************************************************************
 * Notas de Modelos Predictivos;
 * Jose Enrique Perez ;
 * Maestría en Ciencias Actuariales;
 * Facultad de Negocios. Universidad La Salle México;
 * Purpose: To classify images with a neural network.
 **********************************************************************/

/*****************************************************************************/
/*  Terminate the specified CAS session (mySession). No reconnect is possible*/
/*  Run Terminate below if necessary to reset the session.					 */
/*****************************************************************************/
cas casauto terminate;
/* To make a new CAS library */
libname mycas cas;

/*
Checking the features of the server where SAS is running
*/

* Number of CPU;
%put &=sysncpu.;
* Features of your SAS software;
proc setinit; 
run;

/***************************************************************/
/* Load the image actionset and create a metadata table 	   */
/* describing the images.							           */
/* The LargetrainData contains  		  					   */
/* ten sub folders. Each folder contains images of a specific  */
/* image class.      										   */
/* The recurse argument will read in all the images from the   */
/* ten directories and the labelLevels argument will label     */
/* the images according to each subdirectory to keep the data  */
/* organized.  												   */
/***************************************************************/

/* Change the path in according to your SAS Viya session */
/* It creates a library referring to the path where the images are seved */
proc cas;
   	loadactionset 'table';
	loadactionset 'image';
	table.addCaslib / name='imagelib' path='/shared/home/perez-jose@lasallistas.org.mx/ModelosPredictivos/Image_Data/' subdirectories=true;
	image.loadimages / caslib='imagelib' path='LargetrainData' recurse=true labellevels=1 decode=true casout={name='LargetrainData', replace=true};
quit;

/************************************/
/* Use PROC PARTITION to partition  */
/* each folder of images into train */
/* and validation data partitions.  */
/************************************/

proc partition data=mycas.LargetrainData samppct=80 samppct2=20 seed=2023 partind;
     by _label_;
     output out=mycas.LargeImageData;
run;

/******************************/
/* Use the shuffle action to  */
/* randomly sort the data.	  */
/******************************/

proc cas;
 table.shuffle / table='LargeImageData' 
 				 casout={name='LargeImageDatashuffled', replace=1};
quit;

/******************************/
/* Print some images  		  */
/******************************/

* Macro variable to control the number of images to display, it must be lower than 500;
%let d=10;

data _null_;
	set mycas.LargeImageDatashuffled
	(where=(
		  _id_<=1+&d. AND _id_>=1 or 
		  _id_<=501+&d. AND _id_>=501 or
		  _id_<=1001+&d. AND _id_>=1001 or
		  _id_<=1501+&d. AND _id_>=1501 or
		  _id_<=2001+&d. AND _id_>=2001 or
		  _id_<=2501+&d. AND _id_>=2501 or
		  _id_<=3001+&d. AND _id_>=3001 or
		  _id_<=3501+&d. AND _id_>=3501 or
		  _id_<=4001+&d. AND _id_>=4001 or
		  _id_<=4501+&d. AND _id_>=4501) 
			keep=_path_ _id_ _label_)
	end=eof;
	if _n_=1 then
		do;
			dcl odsout obj();
			obj.layout_gridded(columns:8);
		end;
	obj.region();
	obj.format_text(text: _label_, just: "c", style_attr: 'font_size=8pt');
	obj.image(file: _path_, width: "112", height: "112");

	if eof then
		do;
			obj.layout_end();
		end;
run;


/*************************************/
/* Summarize the training image data */
/*************************************/

proc cas;
	image.summarizeimages / table={name='LargeImageDatashuffled', where='_PartInd_=1'};
quit;


/*****************************/
/* Build a model shell		 */
/*****************************/

* Documentation of the DeepLearn action set ;
* https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.3/casdlpg/cas-deeplearn-TblOfActions.htm ;

proc cas;
	loadactionset 'DeepLearn';
	
	BuildModel / modeltable={name='ConVNN', replace=1} type = 'CNN' nthreads=&sysncpu.;

	/* Add an input layer		 */
	AddLayer / model='ConVNN' name='data' layer={type='input' nchannels=3 width=32 height=32 randomFlip='H' randomMutation='Random' offsets={113.852228,123.021097,125.294747}}; 
	
	
	/* Add several Convolutional layers */
	AddLayer / model='ConVNN' name='ConVLayer1a' layer={type='CONVO' nFilters=12  width=1 height=1 stride=1 act='ELU'} srcLayers={'data'};
	AddLayer / model='ConVNN' name='ConVLayer1b' layer={type='CONVO' nFilters=12  width=3 height=3 stride=1 act='ELU'} srcLayers={'data'};
	AddLayer / model='ConVNN' name='ConVLayer1c' layer={type='CONVO' nFilters=12  width=5 height=5 stride=1 act='ELU'} srcLayers={'data'};
	AddLayer / model='ConVNN' name='ConVLayer1d' layer={type='CONVO' nFilters=12  width=7 height=7 stride=1 act='ELU'} srcLayers={'data'};
	AddLayer / model='ConVNN' name='ConVLayer1e' layer={type='CONVO' nFilters=16  width=4 height=4 stride=2 dropout=.2 act='ELU'} srcLayers={'data'};
	AddLayer / model='ConVNN' name='ConVLayer1f' layer={type='CONVO' nFilters=16  width=6 height=6 stride=4 dropout=.25 act='ELU'} srcLayers={'data'};
	
	/* Add a concatenation  layer */
	AddLayer / model='ConVNN' name='concatlayer1a' layer={type='concat'} srcLayers={'ConVLayer1a','ConVLayer1b','ConVLayer1c','ConVLayer1d'}; 
	
	/* Add a max pooling layer */
	AddLayer / model='ConVNN' name='PoolLayer1max' layer={type='POOL'  width=2 height=2 stride=2 pool='max'} srcLayers={'concatlayer1a'}; 
	
	/* Add a concatenation  layer */
	AddLayer / model='ConVNN' name='concatlayer2' layer={type='concat'} srcLayers={'PoolLayer1max','ConVLayer1e'}; 
	
	/* Add a max pooling layer */
	AddLayer / model='ConVNN' name='PoolLayer2max' layer={type='POOL'  width=2 height=2 stride=2 pool='max'} srcLayers={'concatlayer2'}; 
	
	/* Add a concatenation  layer */
	AddLayer / model='ConVNN' name='concatlayer3' layer={type='concat'} srcLayers={'PoolLayer2max','ConVLayer1f'}; 
	
	/* Add a max pooling layer */
	AddLayer / model='ConVNN' name='PoolLayer3max' layer={type='POOL'  width=2 height=2 stride=2 pool='max'} srcLayers={'concatlayer3'}; 
	
	/* Add a Convolutional layer with 64, 3 by 3 filters, a stride of 2 and batch normalization */
	AddLayer / model='ConVNN' name='ConVLayer2a' layer={type='CONVO' nFilters=64 width=3 height=3 stride=2 act='Identity' } srcLayers={'concatlayer3'}; 
	AddLayer / model='ConVNN' name='BatchLayer2a' layer={type='BATCHNORM' act='ELU'} srcLayers={'ConVLayer2a'}; 
	
	/* Add a Convolutional layer with 64, 1 by 1 filters, a stride of 1 and batch normalization */
	AddLayer / model='ConVNN' name='ConVLayer2b' layer={type='CONVO' nFilters=64  width=1 height=1 stride=1 act='Identity' } srcLayers={'concatlayer3'};
	AddLayer / model='ConVNN' name='BatchLayer2b' layer={type='BATCHNORM' act='ELU'} srcLayers={'ConVLayer2b'}; 
	
	/* Add a Convolutional layer with 64, 3 by 3 filters, a stride of 1 and batch normalization */
	AddLayer / model='ConVNN' name='ConVLayer3a' layer={type='CONVO' nFilters=64 width=3 height=3 stride=1 init='msra' act='Identity'} srcLayers={'BatchLayer2b'}; 
	AddLayer / model='ConVNN' name='BatchLayer3a' layer={type='BATCHNORM' act='ELU'} srcLayers={'ConVLayer3a'}; 
	
	/* Add a Convolutional layer with Batch Normalization */
	AddLayer / model='ConVNN' name='ConVLayer3b' layer={type='CONVO' nFilters=64 width=5 height=5 stride=1 init='msra' act='Identity' } srcLayers={'BatchLayer2b'}; 
	AddLayer / model='ConVNN' name='BatchLayer3b' layer={type='BATCHNORM' act='ELU'} srcLayers={'ConVLayer3b'}; 
	
	/* Add a concatenation  layer */
	AddLayer / model='ConVNN' name='concatlayer4' layer={type='concat'} srcLayers={'BatchLayer3a','BatchLayer3b'}; 
	
	/* Add a Convolutional layer with Batch Normalization */
	AddLayer / model='ConVNN' name='ConVLayer4' layer={type='CONVO' nFilters=128  width=3 height=3 stride=2 init='msra2' act='Identity'} srcLayers={'concatlayer4'}; 
	AddLayer / model='ConVNN' name='BatchLayer4' layer={type='BATCHNORM' act='ELU'} srcLayers={'ConVLayer4'}; 
	
	/* Add a concatenation  layer */
	AddLayer / model='ConVNN' name='concatlayer5' layer={type='concat'} srcLayers={'PoolLayer3max','BatchLayer4','BatchLayer2a'}; 
	
	/* Add a Convolutional layer with Batch Normalization */
	AddLayer / model='ConVNN' name='ConVLayerLasta' layer={type='CONVO' nFilters=500  width=1 height=1 stride=1 init='msra2' act='Identity'} srcLayers={'concatlayer5'}; 
	AddLayer / model='ConVNN' name='BatchLayerLasta' layer={type='BATCHNORM' act='ELU'} srcLayers={'ConVLayerLasta'}; 
	
	/* Add a fully-connected layer with Batch Normalization */
	AddLayer / model='ConVNN' name='FCLayer1' layer={type='FULLCONNECT' n=540 act='Identity' init='msra2'}  srcLayers={'BatchLayerLasta'};  
	AddLayer / model='ConVNN' name='BatchLayerFC1' layer={type='BATCHNORM' act='ELU'} srcLayers={'FCLayer1'};
	
	/* Add a fully-connected layer with Batch Normalization */
	AddLayer / model='ConVNN' name='FCLayer2' layer={type='FULLCONNECT' n=540 act='Identity' init='msra2' dropout=.7}  srcLayers={'BatchLayerFC1'};  
	AddLayer / model='ConVNN' name='BatchLayerFC2' layer={type='BATCHNORM' act='ELU'} srcLayers={'FCLayer2'};
	
	/* Add an output layer with softmax activation */
	AddLayer / model='ConVNN' name='outlayer' layer={type='output' act='SOFTMAX'} srcLayers={'BatchLayerFC2'};

quit;

/****************************************/
/* Train the CNN model, ConVNN			*/
/****************************************/
ods output OptIterHistory=ObjectModeliter;
proc cas;
	dlTrain / table={name='LargeImageDatashuffled', where='_PartInd_=1'} model='ConVNN' 
        modelWeights={name='ConVTrainedWeights_d', replace=1}
        bestweights={name='ConVbestweights', replace=1}
        inputs='_image_' 
		nthreads = &sysncpu.
        target='_label_' nominal={'_label_'}
        ValidTable={name='LargeImageDatashuffled', where='_PartInd_=2'} 
        optimizer={minibatchsize=80, 
        			algorithm={method='ADAM', lrpolicy='Step', gamma=0.6, stepsize=10,
       							beta1=0.9, beta2=0.999, learningrate=.01}
        			maxepochs=60} 
        seed=2023;
quit;

/****************************************/
/* Score the data set					*/
/****************************************/
proc cas;
	dlScore / initWeights={name="ConVTrainedWeights_d"}
      	modelTable={name="ConVNN"}
		copyVars={"_id_", "_label_", "_path_"}
      	table={name="LargeImageDatashuffled"}
      	casOut={name="train_scored",replace=TRUE};
quit;

/* Confusion matrix */
proc freq data=mycas.train_scored;
	table _label_*_DL_predname_;
run;

* Checking some scores with images;
%let d=10;

data _null_;
	set mycas.train_scored
	(where=(
		  _id_<=1+&d. AND _id_>=1 or 
		  _id_<=501+&d. AND _id_>=501 or
		  _id_<=1001+&d. AND _id_>=1001 or
		  _id_<=1501+&d. AND _id_>=1501 or
		  _id_<=2001+&d. AND _id_>=2001 or
		  _id_<=2501+&d. AND _id_>=2501 or
		  _id_<=3001+&d. AND _id_>=3001 or
		  _id_<=3501+&d. AND _id_>=3501 or
		  _id_<=4001+&d. AND _id_>=4001 or
		  _id_<=4501+&d. AND _id_>=4501) 
			keep=_path_ _id_ _DL_predname_)
	end=eof;
	if _n_=1 then
		do;
			dcl odsout obj();
			obj.layout_gridded(columns:8);
		end;
	obj.region();
	obj.format_text(text: _DL_predname_, just: "c", style_attr: 'font_size=8pt');
	obj.image(file: _path_, width: "112", height: "112");

	if eof then
		do;
			obj.layout_end();
		end;
run;

/****************************/
/*  Store minimum training  */
/*  and validation error in */
/*  macro variables. 	    */
/****************************/

proc sql noprint;
	select min(FitError)
	into :Train separated by ' '
	from ObjectModeliter;
quit;

proc sql noprint; 
	select min(ValidError)
 	into :Valid separated by ' ' 
	from ObjectModeliter; 
quit; 

/* Plot Performance */
proc sgplot data=ObjectModeliter;
	yaxis label='Misclassification Rate' MAX=.9 min=0;
	series x=Epoch y=FitError / CURVELABEL="Error = &Train. (Training)" CURVELABELPOS=END;
   	series x=Epoch y=ValidError / CURVELABEL="Error = &Valid. (Validation)" CURVELABELPOS=END; 
run;







