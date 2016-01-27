
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

					if length(LRsub.kt) > 1
						nrms = 10*std(LRsub.kt);
					else
						nrms = LRsub.kt;
					end
					for ii = 1:LRsub.rank
						LRsub.kt(:,ii) = LRsub.kt(:,ii)/nrms(ii);
						LRsub.ksp(:,ii) = LRsub.ksp(:,ii)*nrms(ii);
					end
					LRsub.filtK = LRsub.kt * LRsub.ksp';
					LRsub.filtK = LRsub.filtK(:);
				end

				%%
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
					
%% DISPLAY FUNCTIONS
	function [] = display_filter( subunit, dims, varargin )
	% Usage: [] = subunit.display_filter( dims, <plot_location>, varargin )
	%
	% Plots low-rank subunit filter in a 2-row, 1-column subplot
	% INPUTS:
	%	  plot_location: 3-integer list = [Fig_rows Fig_col Loc] arguments to subplot. Default = [1 2 1]
	%	  optional arguments (varargin)
	%	    'color': enter to specify color order of non-image-plots (default is 'bcgrm...').
	%			'colormap': choose colormap for 2-D plots. Default is 'gray'
	%	    'dt': enter if you want time axis scaled by dt
	%	    'time_rev': plot temporal filter reversed in time (zero lag on right)
	%	    'xt_rev': plot 2-D plots with time on x-axis
	%	    'notitle': suppress title labeling subunit type
	
		assert((nargin > 1) && ~isempty(dims), 'Must enter filter dimensions.' )
		[plotloc,parsed_options] = NIM.parse_varargin( varargin );
		if isempty(plotloc)
			plotloc = [1 2 1];
		end
		assert(plotloc(3) <= prod(plotloc(1:2)),'Invalid plot location.')
		if isfield(parsed_options,'color')
			clrs = parsed_options.color;
		else
			clrs = 'bcgrm';
		end
		Nclrs = length(clrs);
		
		if isfield(parsed_options,'colormap')
			clrmap = parsed_options.colormap;
		else
			clrmap = 'gray';
		end
		
		% Time axis details
		NT = dims(1);
		if isfield(parsed_options,'dt')
			dt = parsed_options.dt;
		else
			dt = 1;
		end
		if isfield(parsed_options,'time_rev')
			ts = -dt*(0:NT-1);
		else
			ts = dt*(1:NT);
			if isfield(parsed_options,'dt')
				ts = ts-dt;  % if entering dt into time axis, shift to zero lag
			end
		end

		L = max(subunit.kt(:))-min(subunit.kt(:));

		% Plot temporal filter(s)
		subplot( plotloc(1), plotloc(2), plotloc(3) ); hold on
		for nn = 1:subunit.rank
			plot( ts, subunit.kt(:,nn), clrs(mod(nn-1,Nclrs)+1), 'LineWidth',0.8 );
			axis([min(ts) max(ts) min(subunit.kt(:))+L*[-0.1 1.1]])
			if isfield(parsed_options,'time_rev')
				box on
			else
				box off
			end
		end
		plot( [ts(1) ts(end)],[0 0],'k' )  % plot axis
		
		% Add title
		if ~isfield(parsed_options,'notitle')
			if strcmp(subunit.NLtype,'lin')
				title(sprintf('Lin'),'fontsize',10);
			elseif subunit.weight == 1
				title(sprintf('Exc'),'fontsize',10);
			elseif subunit.weight == -1				
				title(sprintf('Sup'),'fontsize',10);			
			end	
		end
		
		% Plot spatial filter(s)
		subplot( plotloc(1), plotloc(2), plotloc(3)+1 ); 
		if dims(3) == 1

			% then 1-d spatial
			hold on
			for nn = 1:subunit.rank
				plot( subunit.ksp(:,nn), clrs(mod(nn-1,Nclrs)+1), 'LineWidth',0.8 );
			end
			plot( [1 dims(2)],[0 0],'k' )  % plot axis
			L = max(subunit.ksp(:))-min(subunit.ksp(:));
			axis([1 dims(2) min(subunit.ksp(:))+L*[-0.1 1.1]])
		else	
			
			% 2-d spatial
			if subunit.rank > 1
				warning( 'This function will only display first spatial filter.' )
			end
			imagesc( reshape(subunit.ksp,dims(2:3)), max(abs(subunit.ksp(:)))*[-1 1] )
			colormap(clrmap)
			if dims(2) == dims(3)
				axis square
			end
		end
	end

end
end

