%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Organization before main ANN process
%
%                                                  Written by Kim, Wiback,
%                                                              2016.03.18.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%% Pre-processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%
% Clearing & Loading
%%%%%%%%%%%%%%%%%%%%
clear; close all; clc;
% Printing
fprintf('Loading original data. Please wait...\n');
% Start the loading.
load('TIMIT_DR1.mat')
load('TIMIT_DR2.mat')
load('TIMIT_DR3.mat')
load('TIMIT_DR4.mat')
load('TIMIT_DR5.mat')
load('TIMIT_DR6.mat')
load('TIMIT_DR7.mat')
load('TIMIT_DR8.mat')
TIMIT.TRAIN.DR1 = DR_1;
TIMIT.TRAIN.DR2 = DR_2;
TIMIT.TRAIN.DR3 = DR_3;
TIMIT.TRAIN.DR4 = DR_4;
TIMIT.TRAIN.DR5 = DR_5;
TIMIT.TRAIN.DR6 = DR_6;
TIMIT.TRAIN.DR7 = DR_7;
TIMIT.TRAIN.DR8 = DR_8;



%%%%%%%%%%%%%%%%%%%%%
% Data to cell arrays
%%%%%%%%%%%%%%%%%%%%%
dr_1 = struct2cell(TIMIT.TRAIN.DR1);
dr_2 = struct2cell(TIMIT.TRAIN.DR2);
dr_3 = struct2cell(TIMIT.TRAIN.DR3);
dr_4 = struct2cell(TIMIT.TRAIN.DR4);
dr_5 = struct2cell(TIMIT.TRAIN.DR5);
dr_6 = struct2cell(TIMIT.TRAIN.DR6);
dr_7 = struct2cell(TIMIT.TRAIN.DR7);
dr_8 = struct2cell(TIMIT.TRAIN.DR8);



%%%%%%%%%%%%
% Extracting
%%%%%%%%%%%%
% Calling extract.m
[end_1, end_2, organized] = extract(dr_1, 1, 1);
[end_1, end_2, organized] = extract(dr_2, end_1, end_2, organized);
[end_1, end_2, organized] = extract(dr_3, end_1, end_2, organized);
[end_1, end_2, organized] = extract(dr_4, end_1, end_2, organized);
[end_1, end_2, organized] = extract(dr_5, end_1, end_2, organized);
[end_1, end_2, organized] = extract(dr_6, end_1, end_2, organized);
[end_1, end_2, organized] = extract(dr_7, end_1, end_2, organized);
[end_1, end_2, organized] = extract(dr_8, end_1, end_2, organized);





%% MFCC extraction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%
% Parameters
%%%%%%%%%%%%
fs = 16000;
% Window
TW = 25;
% Ovelap between the windows
TS = 10;
% Pre emphasis coefficient, aka amplifier
alpha = 0.97;
% Hamming window function, this is a convention.
hamming = @(N)(0.54 - 0.46 * cos(2 * pi * (0:N - 1).' / (N - 1)));
% Frequency range to consider
R = [300, 3700];
% Number of filterbank channels
M = 24;
% Number of cepstral coefficients
N = 24;
% Cepstral sine lifter parameter
L = 22;



%%%%%%%%%%%%%%%%%
% Extraction loop
%%%%%%%%%%%%%%%%%
print_index = 1;
for n = 1:length(organized)
    % Calling mfcc_rev.m
    [CC, FBE, frames, center_time] = ...
        mfcc_rev(organized(n).signal, fs, TW, TS, ...
        alpha, hamming, R, M, N, L );
    % Creating 1st, 2nd derivations
    first_deriv = diff(CC, 1, 2);
    second_deriv = diff(CC, 2, 2);
    % Padding vacant spaces (due to nature of derivation)
    CC(25:72, 1) = 0;
    % Inserting the 1st, 2nd derivations
    CC(25:48, 2:end) = first_deriv;
    CC(49:72, 3:end) = second_deriv;
    % Concatenating to the structure
    organized(n).MFCC = CC;
    
    %%% Printing results
    if n == 100 * print_index
        print_index = print_index + 1;
        fprintf('MFCC extraction: (%d/%d)\n', n, length(organized));
    elseif n == length(organized)
        fprintf('Done!\n');
    end
end





%% Phones %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
phones_list_without_epi_pau = {'b', 'd', 'g', 'p', 't', 'k', 'jh', ...
    'ch', 's', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh', 'm', 'n', ...
    'ng', 'l', 'r', 'w', 'y', 'hh', 'iy', 'ih', 'eh', 'ey', 'ae', ...
    'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'er', ...
    'ax', 'h#'};



%%%%%%%%%%%%%%%%%%%%%%
% epi + pau + h# == h#
%%%%%%%%%%%%%%%%%%%%%%
% Dummy index for main loop below
print_index = 1;
for n = 1:length(organized)
    for m = 1:length(organized(n).phone)
        % Dummy temp variable for re-allocating
        temp = strrep(organized(n).phone(m), 'epi', 'h#');
        organized(n).phone(m) = temp;
        % Dummy temp variable for re-allocating
        temp = strrep(organized(n).phone(m), 'pau', 'h#');
        organized(n).phone(m) = temp;
        
        %%% Printing results
        if n == 100 * print_index
            print_index = print_index + 1;
            fprintf('Merging phones: (%d/%d)\n', n, length(organized));
        % Print when the loop hits very last sample.
        elseif n == length(organized) && m == length(organized(n).phone)
            fprintf('Done!\n');
        end
    end
end





%% Inputs & Outputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pre-allocating for better speed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_of_columns = 0;

%%% Dumping all unnecessary phones
for n = 1:length(organized)
    for m = 1:length(organized(n).phone)
        for p = 1:length(phones_list_without_epi_pau)
            % Number of valid phones as number of columns
            if strcmp(organized(n).phone{m}, ...
                    phones_list_without_epi_pau{p})
                % The number of columns increses with the each valid match.
                num_of_columns = num_of_columns + 1;
            end
        end
    end
end
% All dummies are sharing the same number of columns for sync.
inputs(size(organized(1).MFCC, 1) * 11, num_of_columns) = 0;
outputs(length(phones_list_without_epi_pau), num_of_columns) = 0;
phone_id_indices(1, num_of_columns) = 0;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creating the inputs & outputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dummy indices for main loop below
input_index = 1;
output_index = 1;
phone_id_index = 1;
print_index = 1;
for n = 1:length(organized)
    % Calculating samples per MFCC frame
    samples_per_mfcc = max(organized(n).phone_sample{end}) / ...
        max(size(organized(n).MFCC));
    % Proceed only with the valid phones.
    for m = 1:length(organized(n).phone)
        for p = 1:length(phones_list_without_epi_pau)
            % Checking whether a phone is valid or not
            if strcmp(organized(n).phone{m}, ...
                    phones_list_without_epi_pau{p})
                % Get medians of each phone, and corresponding MFCC frames.
                minimum = min(organized(n).phone_sample{m});
                maximum = max(organized(n).phone_sample{m});
                % This is the median sample point.
                median = (minimum + maximum) / 2;
                % This is the corresponding MFCC median-ish point.
                mfcc_median_frame = round(median / ...
                    samples_per_mfcc, 0);
                % Five frames to left
                minus_5 = mfcc_median_frame - 5;
                % Five frames to right
                plus_5 = mfcc_median_frame + 5;
                
                %%% Creating the inputs
                % No MFCC column number can be lower than 1.
                if minus_5 < 1
                    % If so, just extract 1 ~ 11 columns (frames).
                    inputs(:, input_index) = ...
                        reshape(organized(n).MFCC(:, 1:11), ...
                        [numel(organized(n).MFCC(:, 1:11)), 1]);
                    input_index = input_index + 1;
                % No MFCC column number can go beyond the size of the data.
                elseif plus_5 > length(organized(n).phone_sample)
                    % If so, just extract last eleven columns (frames).
                    inputs(:, input_index) = ...
                        reshape(organized(n).MFCC(:, end-10:end), ...
                        [numel(organized(n).MFCC(:, end-10:end)), 1]);
                    input_index = input_index + 1;
                % With other cases, follow the convention (5 + median + 5).
                else
                    inputs(:, input_index) = ...
                        reshape(organized(n).MFCC(:, minus_5:plus_5), ...
                        [numel(organized(n).MFCC(:, minus_5:plus_5)), 1]);
                    input_index = input_index + 1;
                end
                
                %%% Creating the outputs
                phone_id = zeros(41, 1);
                phone_id(p) = 1;
                outputs(:, output_index) = phone_id;
                output_index = output_index + 1;
                
                %%% Saving phone indices for later use
                phone_id_indices(phone_id_index) = p;
                phone_id_index = phone_id_index + 1;
            end
        end
    end
    
    %%% Printing results
    if n == 100 * print_index
        print_index = print_index + 1;
        fprintf('In & Outputs creation: (%d/%d)\n', n, length(organized));
    % Print when the loop hits very last sample.
    elseif n == length(organized) && m == length(organized(n).phone)
        fprintf('Done!\n');
    end
end

%%% Clearing for better readability
clearvars -except inputs outputs phone_id_indices ...
    phones_list_without_epi_pau 