
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
       %snim = fit_Tfilters(snim, Robs, Xstims, varargin); %filter model time-filters 
       %snim = fit_Sfilters(snim, Robs, Xstims, varargin); %filter model space-filters
			 % All these will be overloaded
			 snim = fit_filters(snim, Robs, Xstims, varargin); %filter model time-filters 
       %nim = fit_upstreamNLs(nim, Robs, Xstims, varargin); %fit model upstream NLs
       %nim = fit_spkNL(nim, Robs, Xstims, varargin); %fit parameters of spkNL function
       %nim = fit_NLparams(nim, Robs, Xstims, varargin); %fit parameters of (parametric) upstream NL functions
       %nim = fit_weights(nim, Robs, Xstims, varargin); %fit linear weights on each subunit
       [] = display_model(snim,Robs,Xstims,varargin); %display current model
       %[] = display_model_dab(nim,Robs,Xstims,varargin); %display current model
    end
    methods (Static, Hidden)
        %Tmat = create_Tikhonov_matrix(stim_params, reg_type); %make regularization matrices
        %Xmat = create_time_embedding(stim,params) %make time-embedded stimulus
    end
    %%
    methods
        %% CONSTRUCTOR
				function snim = sNIM( nim, ranks, stim_params )
%         snim = sNIM( nim, rank(s) ) or snim = sNIM( STA, rank(s), stim_params )
%         the second (sta) usage defaults to creating an LN model with STA as first filter
%         constructor for class sNIM -- must be based on NIM or STA 
%            INPUTS:
%                nim
%           OUTPUTS:
%                snim: initialized model object
                   
            if nargin == 0
                return %handle the no-input-argument case by returning a null model. This is important when initializing arrays of objects
						end
						if (nargin < 2) || isempty(ranks)
							ranks = 1;  % assume separable model if ranks not entered)
						end
						if length(nim) > 1
							% then assume passed in STA and produce NIM for it with second argument
							sta = nim;
							assert( nargin < 3, 'Need to enter stim_params as third constructor argument.' );
							% trim STA to size specified in stim-params (assume number of lags could be off)
							if size(sta,1) > stim_params.dims(1)
								ks{1} = sta(1:stim_params.dims(1),:);
							else
								ks{1} = zeros( stim_params.dims(1), prod(stim_params.dims(2:3)) );
								ks{1}(1:size(sta,1),:) = sta;
							end
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
							snim.subunits = cat(1,snim.subunits,LRSUBUNIT( nim.subunits(nn), nim.stim_params.dims(1), ranks(nn) ) );
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

					LLtol = 0.0002; MAXiter = 10;

					snim = snim.fit_weights( Robs, stims, 'silent', 1 );

					LL = snim.fit_props.LL; LLpast = -1e10;
					%if ~silent
						fprintf( 'Beginning LL = %f\n', LL )
					%end
					iter = 1;
					while (((LL-LLpast) > LLtol) && (iter < MAXiter))
	
						snim = snim.fit_Tfilters( Robs, stims, 'silent', 1 );
						snim = snim.fit_Sfilters( Robs, stims, 'silent', 1 );

						LLpast = LL;
						LL = snim.fit_props.LL;
						iter = iter + 1;

						%if ~silent
							fprintf( '  Iter %2d: LL = %f\n', iter, LL )
						%end
					end
				end

				function snim = fit_STalt( snim, Robs, stims, varargin )
%					Usage: snim = fit_STalt( snim, Robs, stims, varargin )

					LLtol = 0.0002; MAXiter = 10;

					snim = snim.fit_weights( Robs, stims, 'silent', 1 );

					LL = snim.fit_props.LL; LLpast = -1e10;
					%if ~silent
						fprintf( 'Beginning LL = %f\n', LL )
					%end
					iter = 1;
					while (((LL-LLpast) > LLtol) && (iter < MAXiter))
	
						snim = snim.fit_Sfilters( Robs, stims, 'silent', 1 );
						snim = snim.fit_Tfilters( Robs, stims, 'silent', 1 );

						LLpast = LL;
						LL = snim.fit_props.LL;
						iter = iter + 1;

						%if ~silent
							fprintf( '  Iter %2d: LL = %f\n', iter, LL )
						%end
					end
				end
				
				
				%%
				function snim = fit_Tfilters( snim, Robs, stims, varargin )
%					snim = fit_Tfilters( snim, Robs, Xstims, varargin )
%
%
					%if (nargin < 4) || isempty(train_inds)
					%	train_inds = NaN;
					%end
					% Fit thresholds by default
					varargin{end+1} = 'fit_offsets';
					varargin{end+1} = 1;
				
					[nim,Xs] = snim.convert2NIM_time( stims );
					nim = nim.fit_filters( Robs, Xs, varargin );
					
					% Scoop fit filters back into old struct
					nLags = snim.stim_params.dims(1);
					Nmods = length(snim.subunits);
					for nn = 1:Nmods
						snim.subunits(nn).kt = reshape( nim.subunits(nn).filtK, nLags, snim.subunits(nn).rank );
						snim.subunits(nn) = snim.subunits(nn).normalize_kt();
					end
					
					snim.fit_props = nim.fit_props;
					snim.fit_props.fit_type = 'T-filter';
					snim.fit_history = cat( 1, snim.fit_history, snim.fit_props );

				end
		
				%%
				function snim = fit_Sfilters( snim, Robs, stims, varargin )
%					snim = fit_Sfilters( snim, Robs, Xstims, varargin )
%
%
					[nim,Xs] = snim.convert2NIM_space( stims );
					nim = nim.fit_filters( Robs, Xs, varargin );
					
					% Scoop fit filters back into old struct
					NSP = prod(snim.stim_params.dims(2:3));
					Nmods = length(snim.subunits);
					for nn = 1:Nmods
						snim.subunits(nn).ksp = reshape( nim.subunits(nn).filtK, NSP, snim.subunits(nn).rank );
						snim.subunits(nn) = snim.subunits(nn).normalize_kt();
					end
					
					snim.fit_props = nim.fit_props;
					snim.fit_props.fit_type = 'S-filter';
					snim.fit_history = cat( 1, snim.fit_history, snim.fit_props );

				end

				
				%%
				function snim = fit_weights( snim, Robs, stims, varargin )
%	        snim = fit_weights( snim, Robs, stims, varargin )
%					Scales spatial functions to optimal weights

					[nim,Xs] = snim.convert2NIM_time( stims );
					nim = nim.fit_weights( Robs, Xs, varargin );
					
					for nn = 1:length(nim.subunits)
						snim.subunits(nn).ksp = snim.subunits(nn).ksp * abs(nim.subunits(nn).weight);
					end
					
					snim.fit_props = nim.fit_props;
					snim.fit_props.fit_type = 'weights';
					snim.fit_history = cat( 1, snim.fit_history, snim.fit_props );

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
%                    ('lambda_type',lambda_val): first input is a string specifying the type of
%                        regularization (e.g. 'd2t' for temporal smoothness). This must be followed by a
%                        scalar/vector giving the associated lambda value(s).
%                    ('NLparams',NLparams): cell array of upstream NL parameter vectors
%            OUPUTS: nim: new nim object
            
						% Make equivalent NIM and make subunits
						
            if ~iscell(NLtypes) && ischar(NLtypes);
                NLtypes = cellstr(NLtypes); %make NL types a cell array
            end
            nSubs = length(mod_signs); %number of subunits being added
            nStims = length(snim.stim_params);
            Xtargets = ones(nSubs,1); %default Xtargets to 1
            if length(NLtypes) == 1 && nSubs > 1
                NLtypes = repmat(NLtypes,nSubs,1); %if NLtypes is specified as a single string, assume we want this NL for all subunits
            end
            init_filts = cell(nSubs,1);
            NLparams = cell(nSubs,1);
            rnk = 1;
						
            %parse input flags
            j = 1; reg_types = {}; reg_vals = [];
            while j <= length(varargin)
                switch lower(varargin{j})
                    case 'xtargs'
                        Xtargets = varargin{j+1};
                        assert(all(ismember(Xtargets,1:nStims)),'invalid Xtargets specified');
                    case 'rank'
                        rnk = varargin{j+1};
                    case 'init_kt'
                        if ~iscell(varargin{j+1}) %if init_filts are specified as a matrix, make them a cell array
                            init_filts = cell(length(mod_signs),1);
                            for ii = 1:length(mod_signs)
                                init_filts{ii} = varargin{j+1}(:,ii);
                            end
                        else
                            init_filts = varargin{j+1};
                        end
                    case 'nlparams'
                        NLparams = varargin{j+1};
                        assert(iscell(NLparams),'NLparams must be input as cell array');
                    case nim.allowed_reg_types
                        reg_types = cat(1,reg_types,lower(varargin{j}));
                        cur_vals = varargin{j+1};
                        reg_vals = cat(2,reg_vals, cur_vals(:));
                    otherwise
                        error('Invalid input flag');
                end
                j = j + 2;
            end
            if size(reg_vals,1) == 1 %if reg_vals are specified as scalars, assume we want the same for all subuntis
                reg_vals = repmat(reg_vals,nSubs,1);
            end
            
            assert(length(Xtargets) == nSubs,'length of mod_signs and Xtargets must be equal');
            %initialize subunits
            for ii = 1:nSubs %loop initializing subunits (start from last to initialize object array)
                stimD = prod(snim.stim_params(Xtargets(ii)).dims); %dimensionality of the current filter
                if isempty(init_filts{ii})
                    init_filt = randn(stimD,1)/stimD; %initialize fitler coefs with gaussian noise
                else
                    init_filt = init_filts{ii};
                end
                
               %use the regularization parameters from the most similar subunit if we have one,
               %otherwise use default init
               same_Xtarg = find([snim.subunits(:).Xtarg] == Xtargets(ii),1); %find any existing subunits with this same Xtarget
               same_Xtarg_and_NL = same_Xtarg(strcmp(snim.get_NLtypes(same_Xtarg),NLtypes{ii})); %set that also have same NL type
               if ~isempty(same_Xtarg_and_NL) 
                    default_lambdas = snim.subunits(same_Xtarg_and_NL(1)).reg_lambdas;
               elseif ~isempty(same_Xtarg)
                    default_lambdas = snim.subunits(same_Xtarg(1)).reg_lambdas;
               else
                   default_lambdas = [];
							 end
               
							 % make NIM subunit
							 new_sub = SUBUNIT(init_filt, mod_signs(ii), NLtypes{ii},Xtargets(ii),NLparams{ii});
               snim.subunits = cat(1,snim.subunits,LRSUBUNIT( new_sub, snim.stim_params(Xtargets(ii)).dims(1), rnk )); %add new subunit
							 % default new subunit rank to 1
							 if ~isempty(default_lambdas)
                   snim.subunits(end).reg_lambdas = default_lambdas;
               end
               for jj = 1:length(reg_types) %add in user-specified regularization parameters
                   assert(reg_vals(ii,jj) >= 0,'regularization hyperparameters must be non-negative');
                   snim.subunits(end).reg_lambdas.(reg_types{jj}) = reg_vals(ii,jj);
               end               
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
					
						% Translate into NIM and use its process_stimulus
						[nim,Xs] = snim.convert2NIM_time( stims );
						[LL,pred_rate,mod_internals,LL_data] = nim.eval_model( Robs, Xs, varargin );

				end
				
				%function nim = init_spkhist(nim,n_bins,varargin)
				%function nim = init_nonpar_NLs(nim, Xstims, varargin)
				%function [filt_SE,hessMat] = compute_filter_SEs(nim, Robs, Xstims, varargin)
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

					Nmods = length(snim.subunits);
					NIMstim_params = snim.stim_params;
					NIMstim_params.dims(1) = 1;
					NSP = prod(NIMstim_params.dims(2:3));
					NX = NIMstim_params.dims(3);
					NT = size(stims,1)*NIMstim_params.up_fac;
					indx = floor((0:(NT-1))/NIMstim_params.up_fac)+1;
					%nim = snim.rmfields( {'subunits','stim_params'} );
					%nim = snim.stripfields();
					
					for nn = 1:Nmods
						LRsub = snim.subunits(nn);
						rnk = LRsub.rank;
						stim_params_list(nn) = NIMstim_params;
						stim_params_list(nn).dims(3) = NX * rnk;
						filt_list{nn} = snim.subunits(nn).ksp(:);
						if rnk == 1
							Xtmp = conv2( stims(indx,:), LRsub.kt, 'full' );
							Xstims{nn}(:,1:NSP ) = Xtmp(1:NT,:);
						else
							Xstims{nn} = zeros(NT,NSP*rnk);
							for mm = 1:rnk
								Xtmp = conv2( stims(indx,:), LRsub.kt(:,mm), 'full' );
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
					
					Nmods = length(snim.subunits);
					NIMstim_params = snim.stim_params;
					NIMstim_params.dims(2:3) = 1;
					nLags = NIMstim_params.dims(1);
					NT = size(stims,1)*NIMstim_params.up_fac;
					%nim = rmfields(snim,{'subunits','stim_params'});
					%nim = snim.stripfields();
		
					for nn = 1:Nmods
						LRsub = snim.subunits(nn);
						rnk = LRsub.rank;
						stim_params_list(nn) = NIMstim_params;
						stim_params_list(nn).dims(1) = nLags * rnk;
						filt_list{nn} = snim.subunits(nn).kt(:);
						
						if rnk == 1
							Xstims{nn} = NIM.create_time_embedding( stims * LRsub.ksp, NIMstim_params );
						else
							Xstims{nn} = zeros(NT,nLags*rnk);
							for mm = 1:rnk
								Xstims{nn}(:,(mm-1)*nLags+(1:nLags)) = NIM.create_time_embedding( stims * LRsub.ksp(:,mm), NIMstim_params );
							end
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

        function [] = check_inputs( snim, Robs, stims, sub_inds, gain_funs )
            % checks the format of common inputs params
            if nargin < 4
                gain_funs = [];
            end
            if nargin < 5
                sub_inds = nan;
            end
            Nsubs = length(snim.subunits);
            for n = 1:Nsubs %check that stimulus dimensions match
                [NT,filtLen] = size(stims{nim.subunits(n).Xtarg}); %stimulus dimensions
                assert(filtLen == prod(nim.stim_params(snim.subunits(n).Xtarg).dims),'Xstim dims dont match stim_params');
            end
            assert(length(unique(cellfun(@(x) size(x,1),Xstims))) == 1,'Xstim elements need to have same size along first dimension');
            assert(size(Robs,2) == 1,'Robs must be a vector');
            assert(iscell(Xstims),'Xstims must for input as a cell array');
            if ~isempty(gain_funs)
                assert(size(gain_funs,1) == NT & size(gain_funs,2) == Nsubs,'format of gain_funs is incorrect');
            end
            if ~isnan(sub_inds)
                assert(min(sub_inds) > 0 & max(sub_inds) <= NT,'invalid data indices specified');
            end
        end
        
        % function nim = set_subunit_scales(nim,fgint)
        
    end
    
%    methods (Static)
        %function stim_params = create_stim_params(dims,varargin)
%		end
		
    %methods (Static, Hidden)
		methods (Hidden)
        %function optim_params = set_optim_params(optimizer,input_params,silent)
        %function percentiles = my_prctile(x,p)

				function nim = rmfields( snim, fields )
%				This doesn't currently work. Do by hand...
						fieldlist = setdiff( fieldnames(snim), fields );
						for nn = 1:length(fieldlist)
							eval(sprintf('nim.%s = snim.(''%s'');', fieldlist{nn}, fieldlist{nn} ));
						end
				end
				
		
				
		end
		

end
