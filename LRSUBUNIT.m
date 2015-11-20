
classdef LRSUBUNIT < SUBUNIT
    % Class implementing the subunits comprising an NIM model
    %
    % Dan Butts, November 2015
    %%
    properties
				rank;				 % rank of filter representation. rank = 0 means use filtK
        %filtK;       % filter coefficients, [dx1] array where d is the dimensionality of the target stimulus
				kt;          % temporal functions (NT x rank)
				ksp;         % spatial functions (NSP x rank)
        %NLtype;      % upstream nonlinearity type (string)
        %NLparams;    % vector of 'shape' parameters associated with the upstream NL function (for parametric functions)
        %NLoffset;    % scalar offset value added to filter output
        %weight;      % subunit weight (typically +/- 1)
        %Xtarg;       % index of stimulus the subunit filter acts on
        %reg_lambdas; % struct of regularization hyperparameters
        %Ksign_con;   %scalar defining any constraints on the filter coefs [-1 is negative con; +1 is positive con; 0 is no con]
        %TBy;         %tent-basis coefficients
        %TBx;         %tent-basis center positions
    end
    properties (Hidden)
        %TBy_deriv;   %internally stored derivative of tent-basis NL
        %TBparams;    %struct of parameters associated with a 'nonparametric' NL
        %scale;       %SD of the subunit output derived from most-recent fit
  end
    
    %%
    methods
        function LRsubunit = LRSUBUNIT( subunit, nLags, rnk )
            %         LRsubunit = LRSUBUNIT(  subunit, nLags, rank )
            %         constructor for LRSUBUNIT class.
            %             INPUTS:
						%                subunit: NIM subunit class with established "full-rank" filter
            %             OUTPUTS: LRsubunit: LRsubunit object

            if nargin == 0
                return %handle the no-input-argument case by returning a null model. This is important when initializing arrays of objects
						end
						if nargin < 3
							rnk = 1;
						end
						NSP = length(subunit.filtK)/nLags;

						SUBfields = fieldnames(subunit);
						
						LRsubunit.rank = rnk;  % new property
						LRsubunit.filtK = subunit.filtK; % field #1 from subunit
						LRsubunit.kt = zeros(nLags,max([rnk 1])); % new property
						LRsubunit.ksp = zeros(NSP,max([rnk 1])); % new property
						
						% the rest of the fields should stay the same
						for ii = 2:length(SUBfields) %loop over fields of optim_params
							eval(sprintf('LRsubunit.%s = subunit.(''%s'');', SUBfields{ii}, SUBfields{ii}) );
						end
						%LRsubunit.NLtype = subunit.NLtype;
						%LRsubunit.NLparams = subunit.NLparams;
						%LRsubunit.NLoffset = subunit.NLoffset;
						%LRsubunit.weight = subunit.weight;
						%LRsubunit.Xtarg = subunit.Xtarg;
						%LRsubunit.reg_lambdas = subunit.reg_lambdas;
						%LRsubunit.Ksign_con = subunit.Ksign_con;
						%LRsubunit.TBy = subunit.TBy;
						%LRsubunit.TBx = subunit.TBx;
						
						if LRsubunit.reg_lambdas.d2xt > 0
							LRsubunit.reg_lambdas.d2x = LRsubunit.reg_lambdas.d2xt;
							LRsubunit.reg_lambdas.d2t = LRsubunit.reg_lambdas.d2xt;
							LRsubunit.reg_lambdas.d2xt = 0;
						end
							
						% Create low-rank decomposition
						if rnk > 0
							[u,s,v] = svd( reshape(subunit.filtK, [nLags NSP]) );
							%disp(sprintf( ' %7.4f', diag(s(1:rnk,1:rnk)) ))
							LRsubunit.kt = u(:,1:rnk);
							LRsubunit.ksp =  v(:,1:rnk) * s(1:rnk,1:rnk);
							% normalize each temporal function
							LRsubunit = LRsubunit.normalize_kt();
						end
				end
        
				%%
				function LRsub = normalize_kt( LRsub )
%					LRsub = normalize_kt( LRsub )					
%					normalize kt (and scales ksp appropriately) and constructs filtK

					nrms = 10*std(LRsub.kt);
					for ii = 1:LRsub.rank
						LRsub.kt(:,ii) = LRsub.kt(:,ii)/nrms(ii);
						LRsub.ksp(:,ii) = LRsub.ksp(:,ii)*nrms(ii);
					end
					LRsub.filtK = LRsub.kt * LRsub.ksp';
					LRsub.filtK = LRsub.filtK(:);
				end

				function kts = timeshift_kts( LRsub, tshift )
%				  Usage: kts = LRsub.timeshift_kts( tshift )
%						Returns shifted versions of the subunit kts
						
						[nLags,rnk] = size(LRsub.kt);
						kts = zeros( nLags, rnk );
						if tshift < 0
							kts(1:end-tshift) = LRsub.kt(tshift+1:end,:);
						else
							kts(tshift+1:end,:) = LRsub.kt(1:end-tshift,:);
						end
				end
				
				function kts = upsample_kts( LRsub, tbspace )
%				  Usage: kts = LRsub.upsample_kts( tbspacing )
%							Generates temporal filters at full temporal resolution through upsampling 
					  
						[nLags,rnk] = size(LRsub.kt);
						% Create a tent-basis (triangle) filter
						tent_filter = [(1:tbspace)/tbspace 1-(1:tbspace-1)/tbspace]/tbspace;
						kts = zeros( tbspace*(nLags+1), rnk );
						for nn = 1:length(tent_filter)
							kts((0:nLags-1)*tbspace+nn,:) = kts((0:nLags-1)*tbspace+nn,:) + tent_filter(nn) * LRsub.kt;
						end
						kts = kts(2:end-1,:);  % shift off first latency (one lag in the future)
				end
						
		end
end

