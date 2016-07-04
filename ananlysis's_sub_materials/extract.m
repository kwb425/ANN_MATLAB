%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extracting only necessary data
%
%                                                  Written by Kim, Wiback,
%                                                              2016.03.18.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [end_1, end_2, new_TIMIT] = extract(data, start_1, start_2, TIMIT)





%% Column wise extraction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Re-using the outputs created before
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin == 4
    organized_TIMIT = TIMIT;
end
dummy_index_1 = start_1;
dummy_index_2 = start_2;



%%%%%%%%%%%%%%%%
% Dummy settings
%%%%%%%%%%%%%%%%
if nargin < 4
    organized_TIMIT.text = NaN;
    organized_TIMIT.signal = NaN;
    organized_TIMIT.phone = NaN;
    organized_TIMIT.phone_sample = NaN;
end
dummy_cell_1 = {};
dummy_cell_2 = {};



%%%%%%%%%%%%
% Extraction
%%%%%%%%%%%%
for n = 1:length(data)
    for m = 1:length(data{n, 1})
        % Extracting sentences and signals
        organized_TIMIT(dummy_index_1).text = data{n, 1}(m).TXT;
        organized_TIMIT(dummy_index_1).signal = data{n, 1}(m).SIG;
        dummy_index_1 = dummy_index_1 + 1;
        % Temporary saving phones and samples
        for z = 1:length(data{n,1}(m).byPHONE)
            dummy_cell_1{z, 1} = data{n,1}(m).byPHONE(z).PHONE;
            dummy_cell_2{z, 1} = data{n,1}(m).byPHONE(z).SAMPLES;
        end
        % Extracting the phones and the samples
        organized_TIMIT(dummy_index_2).phone = dummy_cell_1;
        organized_TIMIT(dummy_index_2).phone_sample = dummy_cell_2;
        dummy_index_2 = dummy_index_2 + 1;
        % Initializing the dummy cells to prevent any overlap
        dummy_cell_1 = {};
        dummy_cell_2 = {};
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%
% Returning for next use
%%%%%%%%%%%%%%%%%%%%%%%%
end_1 = dummy_index_1;
end_2 = dummy_index_2;
new_TIMIT = organized_TIMIT;