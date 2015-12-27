function [] = display_model( snim, stims, Robs, varargin )
%         [] = snim.display_model( <Robs>, <Xstims>, varargin )
%         Creates a display of the elements of a given NIM
%              INPUTS:
%                   <Robs>: observed spiking data. Needed if you want to utilize a spike-history
%                       filter. Otherwise set as empty
%                   <stims>: Stimulus cell array. Needed if you want to display the distributions of generating signals
%                   optional_flags:
%                         ('xtargs',xtargs): indices of stimuli for which we want to plot the filters
%                         ('sub_inds',sub_inds): set of subunits to plot
%                         'no_spknl': include this flag to suppress plotting of the spkNL
%                         'no_spk_hist': include this flag to suppress plotting of spike history filter
%                         ('gain_funs',gain_funs): if you want the computed generating signals to account for specified gain_funs

if nargin < 2; stims = []; end;
if nargin < 3; Robs = []; end;

Nsubs = length(snim.subunits);

Xtargs = [1:length(snim.stim_params)]; %default plot filters for all stimuli
sub_inds = 1:Nsubs; %default plot all subunits
plot_spkNL = true;
plot_spk_hist = true;
gain_funs = [];
j = 1; %initialize counter after required input args
while j <= length(varargin)
    switch lower(varargin{j})
        case 'xtargs'
            Xtargs = varargin{j+1};
            assert(all(ismember(Xtargs,1:length(snim.stim_params))),'invalid Xtargets specified');
            j = j + 2;
        case 'no_spknl'
            plot_spkNL = false;
            j = j + 1;
        case 'gain_funs'
            gain_funs = varargin{j+1};
            j = j + 2;
        case 'no_spk_hist'
            plot_spk_hist = false;
            j = j + 1;
        case 'sub_inds'
            sub_inds = varargin{j+1};
            j = j + 2;
        otherwise
            error('Invalid input flag');
    end
end

Nextra_col = 0;
spkhstlen = snim.spk_hist.spkhstlen;
if spkhstlen > 0 && (plot_spk_hist || plot_spkNL)
	Xspkhst = create_spkhist_Xmat(Robs,snim.spk_hist.bin_edges);
	Nextra_col = 1;
end
n_hist_bins = 500; %internal parameter determining histogram resolution
if ~isempty(stims)
    [G, ~, gint] = snim.process_stimulus(stims,1:Nsubs,gain_funs);
    G = G + snim.spkNL.theta; %add in constant term
    if spkhstlen > 0 %add in spike history filter output
        G = G + Xspkhst*snim.spk_hist.coefs(:);
    end
else
    G = []; gint = [];
end
if ~isempty(G) && plot_spkNL
	Nextra_col = 1;
end

% CREATE FIGURE SHOWING INDIVIDUAL SUBUNITS
for tt = Xtargs(Xtargs > 0) %loop over stimuli
    cur_subs = sub_inds([snim.subunits(sub_inds).Xtarg] == tt); %set of subunits acting on this stim
    
    if ~isempty(cur_subs)
        fig_handles.stim_filts = figure();
				
				maxrank = 0;
				for imod = 1:length(cur_subs)
					maxrank = max([maxrank snim.subunits(cur_subs(imod)).rank]);
				end
				
        %if snim.stim_params(tt).dims(3) > 1 %if 2-spatial-dimensional stim
            %n_columns = nim.stim_params(tt).dims(1) + 1;
				n_rows = length(cur_subs);
				if (n_rows == 1) && plot_spk_hist && (spkhstlen > 0)
					n_rows = 2;
				end
				if snim.stim_params(tt).dims(3) > 1
					n_columns = 1+maxrank + 1 + Nextra_col;
				else
					n_columns = 2 + 1 + Nextra_col;
				end
				nLags = snim.stim_params(tt).dims(1); %time lags
        dt = snim.stim_params(tt).dt; %time res
        %nPix = squeeze(snim.stim_params(tt).dims(2:end)); %spatial dimensions
        %create filter time lag axis
        if isempty(snim.stim_params(tt).tent_spacing)
            tax = (0:(nLags-1))*dt;
        else
            tax = (0:snim.stim_params(tt).tent_spacing:(nLags-1)*snim.stim_params(tt).tent_spacing)*dt;
        end
        tax = tax * 1000; % put in units of ms
        
        for imod = 1:length(cur_subs)
            cur_sub = snim.subunits(cur_subs(imod));
            
							subplot(n_rows,n_columns,n_columns*(imod-1)+1);
							plot(tax,cur_sub.kt);
							xr = tax([1 end]);
							line(xr,[0 0],'color','k','linestyle','--');
							if diff(xr) > 0
								xlim(xr);
							end
							xlabel('Time lag')
              ylabel('Filter coef');

							if strcmp(cur_sub.NLtype,'lin')
                title('Lin','fontsize',14)
              elseif cur_sub.weight > 0
                title('Exc','fontsize',14);
              elseif cur_sub.weight < 0
                title('Sup','fontsize',14);
							end
							
							cl = max(abs(cur_sub.ksp(:)));
							if snim.stim_params(tt).dims(3) == 1 %if < 2 spatial dimensions
								subplot(n_rows,n_columns,n_columns*(imod-1)+2);
								if prod(snim.stim_params(tt).dims) > 1
									plot(cur_sub.ksp)
									xlim([1 snim.stim_params(cur_sub.Xtarg).dims(2)])
									ylim([-cl cl]*1.05)
								else
									title(sprintf('%f',cur_sub.ksp))
								end
							else
								for nn = 1:cur_sub.rank
									subplot(n_rows,n_columns,n_columns*(imod-1)+nn+1);
									imagesc(reshape(cur_sub.ksp(:,nn),snim.stim_params(cur_sub.Xtarg).dims(2:3)));
									caxis([-cl cl]);
									%colormap(jet);
									colormap(gray);
									if diff(snim.stim_params(cur_sub.Xtarg).dims(2:3)) == 0
										axis square
									end
								end
							end
						
            
            %PLOT UPSTREAM NL
						subplot(n_rows,n_columns,imod*n_columns-Nextra_col);
						if ~isempty(gint) %if computing distribution of filtered stim
                [gendist_y,gendist_x] = hist(gint(:,cur_subs(imod)),n_hist_bins);
                
                % Sometimes the gendistribution has a lot of zeros (dont want to screw up plot)
                [a b] = sort(gendist_y);
                if a(end) > a(end-1)*1.5
                    gendist_y(b(end)) = gendist_y(b(end-1))*1.5;
                end
            else
                gendist_x = linspace(-3,3,n_hist_bins); %otherwise, just pick an arbitrary x-axis to plot the NL
            end
            if strcmp(cur_sub.NLtype,'nonpar')
                cur_modx = cur_sub.TBx; cur_mody = cur_sub.TBy;
            else
                cur_modx = gendist_x; cur_mody = cur_sub.apply_NL(cur_modx);
            end
            cur_xrange = cur_modx([1 end]);
            
            if ~isempty(gint)
								if ~strcmp(cur_sub.NLtype,'lin')
									[ax,h1,h2] = plotyy(cur_modx,cur_mody,gendist_x,gendist_y);
									if strcmp(cur_sub.NLtype,'nonpar')
									  set(h1,'Marker','o');
									end
	                set(h1,'linewidth',1)
								end
								%axis square

                xlim(ax(1),cur_xrange)
                xlim(ax(2),cur_xrange);
                if all(cur_mody == 0)
                    ylim(ax(1),[-1 1]);
                else
                ylim(ax(1),[min(cur_mody) max(cur_mody)]);
                end
                set(ax(2),'ytick',[])
                yl = ylim();
                line([0 0],yl,'color','k','linestyle','--');
                ylabel(ax(1),'Subunit output','fontsize',12);
                %ylabel(ax(2),'Probability','fontsize',12)
								%axis square
            else
                h = plot(cur_modx,cur_mody,'linewidth',1);
                if strcmp(cur_sub.NLtype,'nonpar')
                    set(h,'Marker','o');
                end
                xlim(cur_xrange)
                ylim([min(cur_mody) max(cur_mody)]);
                ylabel('Subunit output','fontsize',12);
								%axis square
            end
            box off
            %xlabel('Internal generating function')
						xlabel('g')
            title('Upstream NL','fontsize',12)
				end	
		end		
end

if ~isempty(G) && plot_spkNL
    %fig_handles.spk_nl = figure();
		subplot(n_rows,n_columns,n_columns)
    n_bins = 1000; %bin resolution for G distribution
    [Gdist_y,Gdist_x] = hist(G,n_hist_bins); %histogram the generating signal
    
    %this is a hack to deal with cases where the threshold linear terms
    %create a min value of G
    if Gdist_y(1) > 2*Gdist_y(2)
        Gdist_y(1) = 1.5*Gdist_y(2);
    end
    
    cur_xrange = Gdist_x([1 end]);
    if strcmp(snim.spkNL.type,'logistic')
        NLx = linspace(cur_xrange(1),cur_xrange(2) + diff(cur_xrange)/2,500);
        cur_xrange = NLx([1 end]);
    else
        NLx = Gdist_x;
    end
    NLy = snim.apply_spkNL(NLx);
    NLy = NLy/snim.stim_params(1).dt; %convert to correct firing rate units
    
    [ax,h1,h2] = plotyy(NLx,NLy,Gdist_x,Gdist_y);
    set(h1,'linewidth',1)
    yr = [min([0 min(NLy)]) max(NLy)];
    xlim(ax(1),cur_xrange)
    xlim(ax(2),cur_xrange);
    ylim(ax(1),yr);
    
    xlabel('G')
    ylabel(ax(1),'Firing rate','fontsize',12);
    %ylabel(ax(2),'Probability','fontsize',12)
    set(ax(2),'ytick',[]);
    title('Spiking NL','fontsize',12)
end

if (snim.spk_hist.spkhstlen > 0) && plot_spk_hist
		subplot(n_columns,n_rows,2*n_columns)
    fig_handles.spk_hist = figure();
    subplot(2,1,1)
    stairs(snim.spk_hist.bin_edges(1:end-1)*snim.stim_params(1).dt,snim.spk_hist.coefs);
    xlim(snim.spk_hist.bin_edges([1 end])*snim.stim_params(1).dt)
    xl = xlim();
    line(xl,[0 0],'color','k','linestyle','--');
    xlabel('Time lag');
    ylabel('Spike history filter')
    title('Spike history term','fontsize',14)
    
    subplot(2,1,2)
    stairs(snim.spk_hist.bin_edges(1:end-1)*snim.stim_params(1).dt,snim.spk_hist.coefs);
    xlim(snim.spk_hist.bin_edges([1 end-1])*snim.stim_params(1).dt)
    set(gca,'xscale','log')
    xl = xlim();
    line(xl,[0 0],'color','k','linestyle','--');
    xlabel('Time lag');
    ylabel('Spike history')
    title('Spk Hist Log-axis','fontsize',14)
end


