classdef sNIM < NIM    
	% Class implementation of separable NIM based on NIM class. 
	% The main difference is the use of LRsubunit class (instead of subunit)
	% where the 'LR' stands for "low-rank", and fit-filter methods around this
	%   
	% Created by Dan Butts, September 2015
    
	%% No additional properties
	properties
%        spkNL;          %struct defining the spiking NL function
%        subunits;       %array of subunit objects <- this will change to array of 'LRsubunits' but same name
%        stim_params;    %struct array of parameters characterizing the stimuli that the model acts on, must have a .dims field
%        noise_dist;     %noise distribution class specifying the noise model
%        spk_hist;       %class defining the spike-history filter properties
%        fit_props;      %struct containing information about model fit evaluations
%        fit_hist;       %struct containing info about history of fitting
	end
    
	properties (Hidden)
        %init_props;         %struct containing details about model initialization
        %allowed_reg_types = {'nld2','d2xt','d2x','d2t','l2','l1'}; %set of allowed regularization types
				%allowed_spkNLs = {'lin','rectpow','exp','softplus','logistic'}; %set of NL functions currently implemented
        %version = '0.2';    %source code version used to generate the model
        %create_on = date;    %date model was generated
        %min_pred_rate = 1e-50; %minimum predicted rate (for non-negative data) to avoid NAN LL values
        %opt_check_FO = 1e-2; %threshold on first-order optimality for fit-checking
	end
    
	%% METHODS DEFINED IN SEPARATE FILES
	methods 
		[] = display_model( snim, stims, Robs, varargin ); %display current model
		%[] = display_Tfilters( snim, Robs, stims, varargin ); %display current model
       %snim = fit_Tfilters(snim, Robs, stims, varargin); %filter model time-filters 
       %snim = fit_Sfilters(snim, Robs, stims, varargin); %filter model space-filters
			 % All these will be overloaded
			 %snim = fit_filters(snim, Robs, Xstims, varargin); %filter model time-filters 
       %nim = fit_upstreamNLs(nim, Robs, Xstims, varargin); %fit model upstream NLs
       %nim = fit_spkNL(nim, Robs, Xstims, varargin); %fit parameters of spkNL function
       %nim = fit_NLparams(nim, Robs, Xstims, varargin); %fit parameters of (parametric) upstream NL functions
       %nim = fit_weights(nim, Robs, Xstims, varargin); %fit linear weights on each subunit
	end
	methods (Static)
		%Xmat = create_time_embedding(stim,params) %make time-embedded stimulus
	end
	methods (Static, Hidden)
        %Tmat = create_Tikhonov_matrix(stim_params, reg_type); %make regularization matrices
 				%snim = correct_spatial_signs(snim);
	end
		
		
	%%
	methods	

			function snim = sNIM( nim, ranks, stim_params )
%       CONSTRUCTOR: snim = sNIM( nim, rank(s) ) or snim = sNIM( STA, rank(s), stim_params )
				%       the second (sta) usage defaults to creating an LN model with STA as first filter
				%       constructor for class sNIM -- must be based on NIM or STA 
				%          INPUTS:
				%              nim
				%         OUTPUTS:
				%              snim: initialized model object
            
				if nargin == 0        
					return %handle the no-input-argument case by returning a null model. This is important when initializing arrays of objects
				end
				if (nargin < 2) || isempty(ranks)
					ranks = 1;  % assume separable model if ranks not entered)
				end
				
				if length(nim) > 1		
					% then assume passed in STA and produce NIM for it with second argument		
					sta = nim;
					assert( nargin > 2, 'Need to enter stim_params as third constructor argument.' );
					% trim STA to size specified in stim-params (assume number of lags could be off)
					if size(sta,1) > stim_params.dims(1)
						ktmp = sta(1:stim_params.dims(1),:);
					else		
						ktmp = zeros( stim_params.dims(1), prod(stim_params.dims(2:3)) );
						ktmp(1:size(sta,1),:) = sta;
					end
					ks{1} = ktmp(:);
					nim = NIM( stim_params, {'lin'}, 1, 'init_filts', ks );					
				end

				% Intepret ranks
				Nsubs = length(nim.subunits);
				if length(ranks) < Nsubs
					ranks(end+1:Nsubs) = ones(Nsubs-length(ranks),1)*ranks(1);
				end
				
				% Now convert to SNIM
				snim.spkNL = nim.spkNL;
				for nn = 1:Nsubs
					snim.subunits = cat(1,snim.subunits,LRSUBUNIT( nim.subunits(nn), nim.stim_params(nim.subunits(nn).Xtarg).dims(1), ranks(nn) ) );
				end
				snim.stim_params = nim.stim_params;
				snim.noise_dist = nim.noise_dist;
				snim.spk_hist = nim.spk_hist;
				snim.fit_props = nim.fit_props;
				snim.fit_history = nim.fit_history;
			end
        
				%function nim = set_reg_params(nim, varargin)
				%function nim = set_stim_params(nim, varargin)
				%function lambdas = get_reg_lambdas(nim, varargin)
				%function filtKs = get_filtKs(nim,sub_inds)
				%function NLtypes = get_NLtypes(nim,sub_inds)
				%function [filt_penalties,NL_penalties] = get_reg_pen(nim,Tmats)

				%% FITTING FUNCTIONS
								
				function snim = fit_TSalt( snim, Robs, stims, varargin )
%					Usage: snim = fit_TSalt( snim, Robs, stims, varargin )

					LLtol = 0.0002; MAXiter = 12;

					if ~isempty(varargin) && (length(varargin) == 1) && iscell(varargin{1})
						varargin = varargin{1};
					end

					% Check to see if silent (for alt function)
					silent = 0;
					if ~isempty(varargin)
						for j = 1:length(varargin)
							if strcmp( varargin{j}, 'silent' )
								silent = varargin{j+1};
							end
						end
					end
					
					varargin{end+1} = 'silent';
					varargin{end+1} = 1;

					snim = snim.fit_weights( Robs, stims, varargin );

					LL = snim.fit_props.LL; LLpast = -1e10;
					if ~silent
						fprintf( 'Beginning LL = %f\n', LL )
					end
					iter = 1;
					while (((LL-LLpast) > LLtol) && (iter < MAXiter))
	
						snim = snim.fit_Tfilters( Robs, stims, varargin );
						snim = snim.fit_Sfilters( Robs, stims, varargin );

						LLpast = LL;
						LL = snim.fit_props.LL;
						iter = iter + 1;

						if ~silent
							fprintf( '  Iter %2d: LL = %f\n', iter, LL )
						end
					end
					snim = snim.correct_spatial_signs();
				end

				function snim = fit_STalt( snim, Robs, stims, varargin )
%					Usage: snim = fit_STalt( snim, Robs, stims, varargin )

					if ~isempty(varargin) && (length(varargin) == 1) && iscell(varargin{1})
						varargin = varargin{1};
					end

					% Check to see if silent (for alt function)
					silent = 0;
					if ~isempty(varargin)
						for j = 1:length(varargin)
							if strcmp( varargin{j}, 'silent' );
								silent = varargin{j+1};
							end
						end
					end
					
					LLtol = 0.0002; MAXiter = 12;
					varargin{end+1} = 'silent';
					varargin{end+1} = 1;

					snim = snim.fit_weights( Robs, stims, varargin );

					LL = snim.fit_props.LL; LLpast = -1e10;
					%if ~silent
						fprintf( 'Beginning LL = %f\n', LL )
					%end
					iter = 1;
					while (((LL-LLpast) > LLtol) && (iter < MAXiter))
	
						snim = snim.fit_Sfilters( Robs, stims, varargin );
						snim = snim.fit_Tfilters( Robs, stims, varargin );

						LLpast = LL;
						LL = snim.fit_props.LL;
						iter = iter + 1;

						%if ~silent
							fprintf( '  Iter %2d: LL = %f\n', iter, LL )
						%end
					end
					
					snim = snim.correct_spatial_signs();
				end
				
				
				%%
				function snim = fit_Tfilters( snim, Robs, stims, varargin )
%					snim = fit_Tfilters( snim, Robs, Xstims, varargin )
%					Fits temporal filters, and implicitly upstream nonlinearity thresholds

					if ~isempty(varargin) && (length(varargin) == 1) && iscell(varargin)
						varargin = varargin{1};
					end

					% Fit thresholds by default
					varargin{end+1} = 'fit_offsets';
					varargin{end+1} = 1;
				
					[nim,Xs] = snim.convert2NIM_time( stims );
					nim = nim.fit_filters( Robs, Xs, varargin );
					
					% Scoop fit filters back into old struct
					Nmods = length(snim.subunits);
					for nn = 1:Nmods
						nLags = nim.stim_params(nim.subunits(nn).Xtarg).dims(1);
						snim.subunits(nn).kt = reshape( nim.subunits(nn).filtK, nLags/snim.subunits(nn).rank, snim.subunits(nn).rank );
						snim.subunits(nn) = snim.subunits(nn).normalize_kt();
						snim.subunits(nn).NLoffset = nim.subunits(nn).NLoffset;
					end
					snim.spkNL = nim.spkNL;
					
					% Record fit results
					snim.fit_props = nim.fit_props;
					snim.fit_props.fit_type = 'T-filter';
					snim.fit_history = cat( 1, snim.fit_history, snim.fit_props );
				end
		
				%%
				function snim = fit_Sfilters( snim, Robs, stims, varargin )
%					snim = fit_Sfilters( snim, Robs, Xstims, varargin )
%					Fits spatial filters, and implicitly upstream nonlinearity thresholds

					if ~isempty(varargin) && (length(varargin) == 1) && iscell(varargin)
						varargin = varargin{1};
					end

					[nim,Xs] = snim.convert2NIM_space( stims );
					nim = nim.fit_filters( Robs, Xs, varargin );
					
					% Scoop fit filters back into old struct
					Nmods = length(snim.subunits);
					for nn = 1:Nmods
						NSP = prod(snim.stim_params(snim.subunits(nn).Xtarg).dims(2:3));
						snim.subunits(nn).ksp = reshape( nim.subunits(nn).filtK, NSP, snim.subunits(nn).rank );
						snim.subunits(nn) = snim.subunits(nn).normalize_kt();
					end
					snim.spkNL = nim.spkNL;
					
					snim.fit_props = nim.fit_props;
					snim.fit_props.fit_type = 'S-filter';
					snim.fit_history = cat( 1, snim.fit_history, snim.fit_props );

				end

				function snim = fit_filters( snim, Robs, stims, varargin )
%					Usage: snim = fit_filters( snim, Robs, stims, varargin )
%
%					This is an overloaded function to prevent being used. Will go with fit_TSalt

					%fprintf( '\nNote that fit_filters is not strictly defined for sNIM.\nThis will run fit_TSalt.\n' )
					if ~isempty(varargin) && (length(varargin) == 1) && iscell(varargin{1})
						varargin = varargin{1};
					end
					
					snim = fit_TSalt( snim, Robs, stims, varargin );
				end
				
				
				%%
				function snim = fit_weights( snim, Robs, stims, varargin )
%	        snim = fit_weights( snim, Robs, stims, varargin )
%					Scales spatial functions to optimal weights

					if ~isempty(varargin) && (length(varargin) == 1) && iscell(varargin)
						varargin = varargin{1};
					end
					
					[nim,Xs] = snim.convert2NIM_time( stims );
					nim = nim.fit_weights( Robs, Xs, varargin );
					
					for nn = 1:length(nim.subunits)
						snim.subunits(nn).ksp = snim.subunits(nn).ksp * abs(nim.subunits(nn).weight);
					end
					snim.spkNL = nim.spkNL;
					
					snim.fit_props = nim.fit_props;
					snim.fit_props.fit_type = 'weights';
					snim.fit_history = cat( 1, snim.fit_history, snim.fit_props );

				end
				
				%%
				function snim = reg_pathT( snim, Robs, stims, Uindx, XVindx, varargin )
%					Usage: snim = reg_pathT( snim, Robs, stims, Uindx, XVindx, varargin )
%
%					varargin options: 
%						'fit_subs': subunits to fit

					varargin{end+1} = 'fit_offsets';
					varargin{end+1} = 1;
					varargin{end+1} = 'lambdaID';
					varargin{end+1} = 'd2t';
				
					[nim,Xs] = snim.convert2NIM_time( stims );
					nim = nim.reg_path( Robs, Xs, Uindx, XVindx, varargin );
					
					% Scoop fit filters back into old struct
					Nmods = length(snim.subunits);
					for nn = 1:Nmods
						nLags = nim.stim_params(nim.subunits(nn).Xtarg).dims(1);
						snim.subunits(nn).kt = reshape( nim.subunits(nn).filtK, nLags/snim.subunits(nn).rank, snim.subunits(nn).rank );
						snim.subunits(nn) = snim.subunits(nn).normalize_kt();
						snim.subunits(nn).NLoffset = nim.subunits(nn).NLoffset;
						snim.subunits(nn).reg_lambdas.d2t = nim.subunits(nn).reg_lambdas.d2t;
					end
					
					snim.fit_props = nim.fit_props;
					snim.fit_props.fit_type = 'T-regularization';
					snim.fit_history = cat( 1, snim.fit_history, nim.fit_props );
				end

				function snim = reg_pathSP( snim, Robs, stims, Uindx, XVindx, varargin )
%					Usage: snim = reg_pathSP( snim, Robs, stims, Uindx, XVindx, varargin )
%
%					varargin options: 
%						'fit_subs': subunits to fit

					varargin{end+1} = 'fit_offsets';
					varargin{end+1} = 1;
					varargin{end+1} = 'lambdaID';
					varargin{end+1} = 'd2x';
				
					[nim,Xs] = snim.convert2NIM_space( stims );
					nim = nim.reg_path( Robs, Xs, Uindx, XVindx, varargin );
					
					% Scoop fit filters back into old struct
					Nmods = length(snim.subunits);
					for nn = 1:Nmods
						NSP = prod(nim.stim_params(nim.subunits(nn).Xtarg).dims(2:3));
						snim.subunits(nn).ksp = reshape( nim.subunits(nn).filtK, NSP/snim.subunits(nn).rank, snim.subunits(nn).rank );
						snim.subunits(nn).NLoffset = nim.subunits(nn).NLoffset;
						snim.subunits(nn).reg_lambdas.d2x = nim.subunits(nn).reg_lambdas.d2x;
					end
					
					snim.fit_props = nim.fit_props;
					snim.fit_props.fit_type = 'SP-regularization';
					snim.fit_history = cat( 1, snim.fit_history, nim.fit_props );
				end

				function snim = reg_path( snim, Robs, stims, Uindx, XVindx, varargin )
					display( 'Note this function does not exist for sNIM. Running reg_pathT' )
					if nargin < 6
						snim = reg_pathT( snim, Robs, stims, Uindx, XVindx );
					else
						snim = reg_pathT( snim, Robs, stims, Uindx, XVindx, varargin );
					end
				end
				
				%%
        function snim = add_subunits(snim,NLtypes,mod_signs,varargin)
%         Usage: snim = snim.add_subunits( NLtypes, mod_signs, varargin)
%         Add subunits to the model with specified properties. Can add multiple subunits in one
%         call. Default is to initialize regularization lambdas to be equal to an existing subunit
%         that acts on the same Xtarg (and has same NL type). Otherwise, they are initialized to 0.
%            INPUTS: 
%                 NLtypes: string or cell array of strings specifying upstream NL types
%														allowed: 'lin','quad','rectlin','rectpow','softplus'
%                 mod_signs: vector of weights associated with each subunit (typically +/- 1)
%                 optional flags:
%                    ('xtargs',xtargs): specify vector of of Xtargets for each added subunit
%                    ('rank',rank): rank of added subunit
%                    ('init_kt',init_kt): cell array of initial temporal filter values for each subunit
%                    ('init_ksp',init_ksp): cell array of initial space filter values for each subunit
%                    ('lambda_type',lambda_val): first input is a string specifying the type of
%                        regularization (e.g. 'd2t' for temporal smoothness). This must be followed by a
%                        scalar/vector giving the associated lambda value(s).
%                    ('NLparams',NLparams): cell array of upstream NL parameter vectors
%            OUPUTS: nim: new nim object

						% Extract SNIM-relevant variables from vargin and pass the rest on
						kts = []; ksps = []; ranks = [];
						counter = 1;	j = 1; 
						modvarargin = {};
						while j <= length(varargin)
							switch lower(varargin{j})
								case 'init_kt'
									kts = varargin{j+1};
								case 'init_ksp'
									ksps = varargin{j+1};
								case 'rank'
									ranks = varargin{j+1};
								otherwise
									modvarargin{counter} = varargin{j}; 
									modvarargin{counter+1} = varargin{j+1}; 
									counter = counter + 2;
							end
							j = j + 2;
						end

						% Make equivalent NIM and make subunits
						Nmods = length(snim.subunits);
						oldranks = ones(Nmods,1);
						for nn = 1:Nmods
							init_filts{nn} = snim.subunits(nn).filtK;
							oldranks(nn) = snim.subunits(nn).rank;
						end
						nim = convert2NIM( snim, snim.stim_params, init_filts, ones(Nmods,1) );
						nim = nim.add_subunits( NLtypes, mod_signs, modvarargin );
						
						Nsubsadded = length(nim.subunits)-length(snim.subunits);
						if isempty(ranks)
							ranks = [oldranks; ones(Nsubsadded,1)*oldranks(1);];
						else
							ranks = [oldranks; ranks;];
						end
						snim2 = sNIM( nim, ranks );
						added_subunits = snim2.subunits(length(snim.subunits)+(1:Nsubsadded));
						if ~isempty(kts)
							for nn = 1:length(kts)
								added_subunits(nn).kt = kts{nn}(:);
							end
						end
						if isempty(ksps)
							for nn = 1:Nsubsadded
								added_subunits(nn).ksp = snim.subunits(1).ksp;
							end
						else
							for nn = 1:length(ksps)
								added_subunits(nn).ksp = ksps{nn}(:);
							end
						end
						
						snim.subunits = cat( 1, snim.subunits, added_subunits );
				end

				function snim = subunit_flip( snim, targets )
%					Usage: cnim = subunit_flip( cnim, <targets> )
	
					Nsubs = length(snim.subunits);
					if nargin < 2
						targets = 1:Nsubs;
					end
					for tar = targets
						snim.subunits(tar).kt = -snim.subunits(tar).kt;
						snim.subunits(tar).ksp = -snim.subunits(tar).ksp;
					end
				end
				
				
				function [Uindx,XVindx] = generate_XVfolds( snim, NTstim, Nfold, XVfolds )
%					Usage: [Uindx,XVindx] = generate_XVfolds( snim, NTstim, <Nfold>, <XVfolds> )
%							Generates Uindx and XVindx to use for fold-Xval.
%					INPUTS:
%						NTstim = size(stim,1);  number of time steps in the stimulus
%						Nfold = number of folds (e.g., 5-fold). Default = 5
%						XVfolds = which folds to set aside for X-val. Can be more than 1. Default = in the middle (3 for 5)
%					OUPUTS:
%						Uindx = indices of design matrix (e.g., X-stim) to use for model fitting
%						XVindx = indices of design matrix (e.g., X-stim) to use for cross-validation
				
					upsampling = snim.stim_params(1).up_fac;
					
					if (nargin < 3) || isempty(Nfold)
						Nfold = 5;
					end
					if (nargin < 4) 
						XVfolds = ceil(Nfold/2);
					end

					NT = NTstim*upsampling;
					XVindx = [];
					for nn = 1:length(XVfolds)
						XVindx = cat(1, XVindx, (floor((XVfolds(nn)-1)*NT/Nfold+1):floor(XVfolds(nn)*NT/Nfold))' );
					end
					Uindx = setdiff((1:NT)',XVindx);
			end
				
				function [LL, pred_rate, mod_internals, LL_data] = eval_model( snim, Robs, stims, varargin )
%					Usage: [LL, pred_rate, mod_internals, LL_data] = eval_model( snim, Robs, stims, varargin )

						if ~isempty(varargin)
							if (length(varargin) == 1) && iscell(varargin{1})
								varargin = varargin{1};
							end
						end

						% Translate into NIM and use its process_stimulus
						[nim,Xs] = snim.convert2NIM_time( stims );
						[LL,pred_rate,mod_internals,LL_data] = nim.eval_model( Robs, Xs, varargin );

				end

				function display_model_dab( snim, stims, Robs, varargin )
%					Usage: Not supported with sNIM -- use display_model(...)
					disp( 'This display function is not supported with sNIM. Using standard' );
					display_model( snim, stims, Robs, varargin ); %display current model
				end
				
				function display_Tfilters( snim, clrscheme )
%					Usage: snim.display_Tfilters( <clrscheme> )
					% For now just display kts of first rank relative to one another
					% Default is exc = blue, inh = red, clrscheme > 0 means cycle through bcgrmk
					
					if nargin < 2
						clrscheme = 0;
					end
					dt = snim.stim_params(1).dt*snim.stim_params(1).tent_spacing; %/snim.stim_params(1).up_fac;
					nLags = snim.stim_params(1).dims(1);
					
					clrs = 'bcgrmkbcgrmk';  Eclrs = 'bcgbcg'; Iclrs = 'rmrmrm'; 
					Nmods = length(snim.subunits);
					Nexc = 0;	Ninh = 0;
					figure; hold on
					for nn = 1:Nmods
						if clrscheme == 0
							if snim.subunits(nn).weight > 0
								Nexc = Nexc + 1;
								clr = Eclrs(Nexc);
							else
								Ninh = Ninh + 1;
								clr = Iclrs(Ninh);
							end	
						else
							clr = clrs(nn);
						end
						plot( (0:nLags-1)*dt,snim.subunits(nn).kt(:,1), clr, 'LineWidth',0.8 );
						if snim.subunits(nn).rank > 1
							plot( (0:nLags-1)*dt, snim.subunits(nn).kt(:,2), sprintf('%c--', clr), 'LineWidth',0.8 )
						end
					end
					plot( [0 (nLags-1)*dt],[0 0],'k','LineWidth',0.5)
					xlim([0 (nLags-1)*dt])
				end
				
				%function nim = init_spkhist(nim,n_bins,varargin)
				%function nim = init_nonpar_NLs(nim, Xstims, varargin)
				%function [filt_SE,hessMat] = compute_filter_SEs(nim, Robs, Xstims, varargin)
	

			function [LLs,LLnulls] = eval_model_reps( snim, RobsR, stims, varargin )
%         Usage: [LLs,LLnulls] = snim.eval_model_reps( RobsR, stims, <eval_inds>, <varargin> )
%													or
%								 [LLs,LLnulls] = snim.eval_model_reps( RspksR, Xstims, <eval_inds>, <varargin> )
%
%         Evaluates the model on the supplied data. In this case RobsR would be a NT x Nreps matrix. Can
%         also pass in a list of spike times with repeats separated by -1

					if ~isempty(find(RobsR < 0, 1))  % then its a list of spike times
						RspksR = RobsR;
						if RspksR(end) > 0
							RspksR(end+1) = -1;
						end
						Rlocs = find(RspksR == -1);
						Nreps = length(Rlocs);
						dt = snim.stim_params(1).dt;
						NT = size(stims,1) * snim.stim_params(1).up_fac;
						RobsR = zeros(NT,Nreps);
						startindx = 1;
						for nn = 1:Nreps
							RobsR(:,nn) = histc( RspksR(startindx:(Rlocs(nn)-1)), (0:(NT-1))*dt );
							startindx = Rlocs(nn) + 1;
						end
					end
						
					Nreps = size(RobsR,2);
					LLs = zeros(Nreps,1); 	LLnulls = zeros(Nreps,1);
					for nn = 1:Nreps
						[LLs(nn),~,~,LLdata] = snim.eval_model( RobsR(:,nn), stims, varargin);
						LLnulls(nn) = LLdata.nullLL;
					end
				end
	end
    
    methods (Hidden)
        %% internal methods        
        function [G, fgint, gint] = process_stimulus( snim, stims, sub_inds, gain_funs )
%         [G, fgint, gint] = process_stimulus( nim, stim, <sub_inds>, <gain_funs> )
%         process the stimulus with the subunits specified in sub_inds
%             INPUTS:
%                 stims: stimulus as cell array
%                 sub_inds: set of subunits to process
%                 gain_funs: temporally modulated gain of each subunit
%             OUTPUTS:
%                 G: summed generating signal
%                 gint: output of each subunit filter
%                 fgint: output of each subunit
            
						if nargin < 4
							gain_funs = [];
						end
						if nargin < 3
							sub_inds = [];
						end
						
						% Translate into NIM and use its process_stimulus
						[nim,Xs] = snim.convert2NIM_time( stims );
						[G,fgint,gint] = nim.process_stimulus( Xs, sub_inds, gain_funs );

        end
                
        % function [LL,norm_fact] = internal_LL(nim,rPred,Robs)
        % function LL_deriv = internal_LL_deriv(nim,rPred,Robs)        
        % function rate = apply_spkNL(nim,gen_signal)
        % function rate_deriv = apply_spkNL_deriv(nim,gen_signal,thresholded_inds)        
        % function rate_grad = spkNL_param_grad(nim,params,x)

				% function Tmats = make_Tikhonov_matrices(nim)
        % function Tmat = make_NL_Tmat(nim)

%% CONVERSION FUNCTIONS
				
				function [nim,Xstims] = convert2NIM_space( snim, stims )
%					[nim,Xstims] = convert2NIM_space( snim, stims )
%					Collapse over temporal dimension to fit spatial-only model (nLags = 1)
%					This will have a different Xstim for each subunit
%            INPUTS: 
%            OUPUTS: 
%								nim: new nim object
%								Xstims: design matrices to pair with NIM

					if ~iscell(stims) % enforce use of cells to list stims going into model
						tmp = stims;
						clear stims
						stims{1} = tmp;
					end
					
					Nmods = length(snim.subunits);
					NT = size(stims{1},1)*snim.stim_params(1).up_fac;
					
					for nn = 1:Nmods
						LRsub = snim.subunits(nn);
						rnk = LRsub.rank;
						indx = floor((0:(NT-1))/snim.stim_params(LRsub.Xtarg).up_fac)+1;
						stim_params_list(nn) = snim.stim_params(LRsub.Xtarg);
						stim_params_list(nn).dims(1) = 1;
						NSP = prod(stim_params_list(nn).dims(2:3));
						NX = stim_params_list(nn).dims(3);
						stim_params_list(nn).dims(3) = NX * rnk;
						filt_list{nn} = snim.subunits(nn).ksp(:);
						% Upsample kt if tent-basis representation
						if ~isempty(stim_params_list(nn).tent_spacing)
							kts = LRsub.upsample_kts( stim_params_list(nn).tent_spacing );
						else
							kts = LRsub.kt;
						end
						if rnk == 1
							Xtmp = conv2( stims{LRsub.Xtarg}(indx,:), kts, 'full' );
							Xstims{nn}(:,1:NSP ) = Xtmp(1:NT,:);
						else
							Xstims{nn} = zeros(NT,NSP*rnk);
							for mm = 1:rnk
								Xtmp = conv2( stims{LRsub.Xtarg}(indx,:), kts(:,mm), 'full' );
								Xstims{nn}(:,(mm-1)*NSP+(1:NSP) ) = Xtmp(1:NT,:);
							end
						end
						%nim.subunits = cat(1, nim.subunits, SUBUNIT( k_init, LRsub.weight, LRsub.NLtype, nn, LRsub.NLoffset, LRsub.NLparams, LRsub.Ksign_con ) );
					end
					nim = snim.convert2NIM( stim_params_list, filt_list, 1:Nmods );
					
					% Remove temporal regularization
					for nn = 1:Nmods
						nim.subunits(nn).reg_lambdas.d2t = 0;
					end						
				end

				%%
				function [nim,Xstims] = convert2NIM_time( snim, stims )
%					[nim,Xstims] = convert2NIM_time( stims )
%					Collapse over sptial dimension to fit temporal-only model (NX = NY = 1)
%					This will have a different Xstim for each subunit
%            INPUTS: 
%            OUPUTS: 
%								nim: new nim object
%								Xstims: design matrices to pair with NIM

					if ~iscell(stims) % enforce use of cells to list stims going into model
						tmp = stims;
						clear stims
						stims{1} = tmp;
					end

					Nmods = length(snim.subunits);
					NT = size(stims{1},1)*snim.stim_params(1).up_fac;  % assuming all same NT
		
					for nn = 1:Nmods
						LRsub = snim.subunits(nn);
						rnk = LRsub.rank;
						stim_params_list(nn) = snim.stim_params(LRsub.Xtarg);
						nLags = stim_params_list(nn).dims(1);
						stim_params_list(nn).dims(2:3) = 1;

						%stim_params_list(nn).dims(1) = nLags * rnk;
						filt_list{nn} = snim.subunits(nn).kt(:);
						
						if rnk == 1
							Xstims{nn} = NIM.create_time_embedding( stims{LRsub.Xtarg} * LRsub.ksp, stim_params_list(nn) );
						else
							Xstims{nn} = zeros(NT,nLags*rnk);
							for mm = 1:rnk
								Xstims{nn}(:,(mm-1)*nLags+(1:nLags)) = NIM.create_time_embedding( stims{LRsub.Xtarg} * LRsub.ksp(:,mm), stim_params_list(nn) );
							end
							stim_params_list(nn).dims(1) = nLags * rnk;
						end
					end
					
					nim = snim.convert2NIM( stim_params_list, filt_list, 1:Nmods );

					% Remove spatial regularization
					for nn = 1:Nmods
						nim.subunits(nn).reg_lambdas.d2x = 0;
					end						

				end
				
				%%
				function nim = convert2NIM( snim, stim_params, init_filts, Xtarg )
%%				Usage: nim = convert2NIM( snim, stim_params, init_filts, Xtarg )
%%
					Nmods = length(snim.subunits);
					SUBsigns = zeros(Nmods,1);
					for nn = 1:Nmods
						NLtypes{nn} = snim.subunits(nn).NLtype;
						SUBsigns(nn) = snim.subunits(nn).weight;
						Ksign_cons(nn) = snim.subunits(nn).Ksign_con;
					end
					
					nim = NIM( stim_params, NLtypes, SUBsigns, 'init_filts', init_filts, 'Xtargets', Xtarg, 'noise_dist', snim.noise_dist, 'Ksign_cons', Ksign_cons );
					% Set the rest of the subunit properties by hand
					nim.spkNL = snim.spkNL;
					for nn = 1:Nmods
						nim.subunits(nn).NLparams = snim.subunits(nn).NLparams;
						nim.subunits(nn).NLoffset = snim.subunits(nn).NLoffset;
						%nim.subunits(nn).weight = snim.subunits(nn).weight;
						nim.subunits(nn).reg_lambdas = snim.subunits(nn).reg_lambdas;
						nim.subunits(nn).TBy = snim.subunits(nn).TBy; 
						nim.subunits(nn).TBx = snim.subunits(nn).TBx; 
					end
				end

%         function [] = check_inputs( snim, Robs, stims, sub_inds, gain_funs )
%             % checks the format of common inputs params
%             if nargin < 4
%                 gain_funs = [];
%             end
%             if nargin < 5
%                 sub_inds = nan;
%             end
%             Nsubs = length(snim.subunits);
%             for n = 1:Nsubs %check that stimulus dimensions match
%                 [NT,filtLen] = size(stims{nim.subunits(n).Xtarg}); %stimulus dimensions
%                 assert(filtLen == prod(nim.stim_params(snim.subunits(n).Xtarg).dims),'Xstim dims dont match stim_params');
%             end
%             assert(length(unique(cellfun(@(x) size(x,1),Xstims))) == 1,'Xstim elements need to have same size along first dimension');
%             assert(size(Robs,2) == 1,'Robs must be a vector');
%             assert(iscell(Xstims),'Xstims must for input as a cell array');
%             if ~isempty(gain_funs)
%                 assert(size(gain_funs,1) == NT & size(gain_funs,2) == Nsubs,'format of gain_funs is incorrect');
%             end
%             if ~isnan(sub_inds)
%                 assert(min(sub_inds) > 0 & max(sub_inds) <= NT,'invalid data indices specified');
%             end
%         end
        
        % function nim = set_subunit_scales(nim,fgint)
        
			function snim = correct_spatial_signs( snim )
%				Usage: snim = snim.correct_spatial_signs()
%					Checks each subunit and rank to make sure spatial maps are all positive

					for nn = 1:length(snim.subunits)
						for mm = 1:length(snim.subunits(nn).rank)
							[~,maxloc] = max(abs(snim.subunits(nn).ksp(:,mm)));
							if snim.subunits(nn).ksp(maxloc,mm) < 0
								snim.subunits(nn).ksp(:,mm) = -snim.subunits(nn).ksp(:,mm);
								snim.subunits(nn).kt(:,mm) = -snim.subunits(nn).kt(:,mm);
							end
						end
					end
			end
		end
		

end
