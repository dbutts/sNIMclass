function sta = spike_triggered_average( Robs, stim, stim_params )
% Usage: sta = spike_triggered_average( Robs, stim, <stim_params> )
%
% Rudimentary spike-triggered average calculation for sNIM code (static function)
% stims and Robs must be formatted in same way as used for model. 'stim_params' optional,
% but defaults will be up_fac =1 (no upsampling) and dims(1) = 50 (# of temporal lags)
%
% This method is not the most efficient, but is less memory-intensive than more efficient ways

%% Establish parameters of STA
[NT,Nsp] = size(stim);
if nargin < 3
	nLags = 50;
	upsampling = 1;
	tent_spacing = 1;
else
	nLags = stim_params.dims(1);
	upsampling = stim_params.up_fac;
	if isempty(upsampling)
		upsampling = 1;
	end
	tent_spacing = stim_params.tent_spacing;
	if isempty(tent_spacing)
		tent_spacing = 1;
	end
	assert(prod(stim_params.dims(2:3)) == Nsp, 'Spatial dimension mismatch.')
end
nLags = nLags * tent_spacing; % will first calculate STA at highest resolution

%% Upsample stim if necessary
if upsampling > 1
	indx = floor((0:(upsampling*NT-1))/upsampling)+1;
	stim = stim(indx,:);
end
assert(length(Robs) == size(stim,1), sprintf( 'Robs is not correct size (NT = %d).', size(stim,1) ))

%% Compute STA
sta = zeros( nLags, Nsp );
% Zero out spikes before nLags
Robs(1:nLags) = 0; 

% Assume spikes are sparse relative to bins and only calculate STA for non-zero responses
spks_pres = find(Robs(:)' > 0);
lags = (0:(nLags-1));

for tt = spks_pres
	sta = sta + Robs(tt) * stim(tt-lags,:);
end

% Normalize by number of spikes used
sta = sta/sum(Robs);

% If tent_spacing > 1, then downsample STA
sta = sta( ceil(tent_spacing/2):tent_spacing:nLags, : );  % chooses middle sample

end
