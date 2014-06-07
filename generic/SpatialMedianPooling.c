#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialMedianPooling.c"
#else

#pragma GCC push_options
#pragma GCC optimize ("unroll-loops")
#pragma GCC optimize ("O4")

static void nn_(SpatialMedianPooling_updateOutput_frame)(real *input_p, real *output_p,
							 real *indx_p, real *indy_p,
							 long nslices,
							 long iwidth, long iheight,
							 long owidth, long oheight,
							 int kW, int kH, int dW, int dH)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    /* loop over output */
    long i, j;
    for(i = 0; i < oheight; i++)
    {
      for(j = 0; j < owidth; j++)
      {
        /* local pointers */
        real *ip = input_p   + k*iwidth*iheight + i*iwidth*dH + j*dW;
        real *op = output_p  + k*owidth*oheight + i*owidth + j;
        real *indyp = indy_p + k*owidth*oheight + i*owidth + j;
        real *indxp = indx_p + k*owidth*oheight + i*owidth + j;

        /* determine the median */
	int M = (kW * kH) / 2; /* position of the median */
	int m;
	/* medianval is the m-th largest entry in the neighborhood patch */
	real medianval = THInf;
	long medianindex = -1;
	/* maxval is the largest value that is smaller than medianval
	   in the current run ("running maximum") */
	real maxval; 
	long maxindex = -1;
        long tcntr;
        int x,y;
	for(m = 0; m <= M; m++)
	{
	  tcntr = 0;
	  maxval = -THInf;
	  for(y = 0; y < kH; y++)
	  {
	    for(x = 0; x < kW; x++)
	    {
	      real val = *(ip + y*iwidth + x);
	      if (val > maxval && val < medianval)
	      {
		maxval = val;
		maxindex = tcntr;
	      }
	      tcntr++;
	    }
	  }
	  /* update the median */
	  medianval = maxval;
	  medianindex = maxindex;
	}

        /* set output to local max */
        *op = medianval;

        /* store location of max (x,y) */
        *indyp = (int)(medianindex / kW)+1;
        *indxp = (medianindex % kW) +1;
      }
    }
  }
}

static int nn_(SpatialMedianPooling_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  long nslices;
  long iheight;
  long iwidth;
  long oheight;
  long owidth;
  real *input_data;
  real *output_data;
  real *indices_data;


  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4 , 2, "3D or 4D (batch mode) tensor expected");

  if (input->nDimension == 4) 
  {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }
  luaL_argcheck(L, input->size[dimw] >= kW && input->size[dimh] >= kH, 2, "input image smaller than kernel size");

  /* sizes */
  nslices = input->size[dimh-1];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];
  oheight = (iheight - kH) / dH + 1;
  owidth = (iwidth - kW) / dW + 1;

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

  /* resize output */
  if (input->nDimension == 3)
  {
    THTensor_(resize3d)(output, nslices, oheight, owidth);
    /* indices will contain i,j locations for each output point */
    THTensor_(resize4d)(indices, 2, nslices, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);

    nn_(SpatialMedianPooling_updateOutput_frame)(input_data, output_data,
						 indices_data+nslices*owidth*oheight, indices_data,
						 nslices,
						 iwidth, iheight,
						 owidth, oheight,
						 kW, kH, dW, dH);
  }
  else
  {
    long p;

    THTensor_(resize4d)(output, nbatch, nslices, oheight, owidth);
    /* indices will contain i,j locations for each output point */
    THTensor_(resize5d)(indices, 2, nbatch, nslices, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      nn_(SpatialMedianPooling_updateOutput_frame)(input_data+p*nslices*iwidth*iheight, output_data+p*nslices*owidth*oheight,
						   indices_data+(p+nbatch)*nslices*owidth*oheight, indices_data+p*nslices*owidth*oheight,
						   nslices,
						   iwidth, iheight,
						   owidth, oheight,
						   kW, kH, dW, dH);
    }
  }

  /* cleanup */
  THTensor_(free)(input);
  return 1;
}

static void nn_(SpatialMedianPooling_updateGradInput_frame)(real *gradInput_p, real *gradOutput_p,
							    real *indx_p, real *indy_p,
							    long nslices,
							    long iwidth, long iheight,
							    long owidth, long oheight,
							    int dW, int dH)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    real *gradInput_p_k = gradInput_p + k*iwidth*iheight;
    real *gradOutput_p_k = gradOutput_p + k*owidth*oheight;
    real *indx_p_k = indx_p + k*owidth*oheight;
    real *indy_p_k = indy_p + k*owidth*oheight;

    /* calculate median points */
    long i, j;
    for(i = 0; i < oheight; i++)
    {
      for(j = 0; j < owidth; j++)
      {
        /* retrieve position of median */
        long maxi = indy_p_k[i*owidth + j] - 1 + i*dH;
        long maxj = indx_p_k[i*owidth + j] - 1 + j*dW;

        /* update gradient */
        gradInput_p_k[maxi*iwidth + maxj] += gradOutput_p_k[i*owidth + j];
      }
    }
  }
}

static int nn_(SpatialMedianPooling_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  int nslices;
  int iheight;
  int iwidth;
  int oheight;
  int owidth;
  real *gradInput_data;
  real *gradOutput_data;
  real *indices_data;

  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  if (input->nDimension == 4) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  /* sizes */
  nslices = input->size[dimh-1];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];
  oheight = gradOutput->size[dimh];
  owidth = gradOutput->size[dimw];

  /* get raw pointers */
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);
  indices_data = THTensor_(data)(indices);

  /* backprop */
  if (input->nDimension == 3)
  {
    nn_(SpatialMedianPooling_updateGradInput_frame)(gradInput_data, gradOutput_data,
						    indices_data+nslices*owidth*oheight, indices_data,
						    nslices,
						    iwidth, iheight,
						    owidth, oheight,
						    dW, dH);
  }
  else
  {
    long p;
#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      nn_(SpatialMedianPooling_updateGradInput_frame)(gradInput_data+p*nslices*iwidth*iheight, gradOutput_data+p*nslices*owidth*oheight,
						      indices_data+(p+nbatch)*nslices*owidth*oheight, indices_data+p*nslices*owidth*oheight,
						      nslices,
						      iwidth, iheight,
						      owidth, oheight,
						      dW, dH);
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);

  return 1;
}

static const struct luaL_Reg nn_(SpatialMedianPooling__) [] = {
  {"SpatialMedianPooling_updateOutput", nn_(SpatialMedianPooling_updateOutput)},
  {"SpatialMedianPooling_updateGradInput", nn_(SpatialMedianPooling_updateGradInput)},
  {NULL, NULL}
};

static void nn_(SpatialMedianPooling_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialMedianPooling__), "nn");
  lua_pop(L,1);
}

#endif
