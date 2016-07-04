%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ANN with TIMIT data
%
%                                                 Written by Kim, Wiback,
%                                                  2016. 03. 18. Ver 1.1.
%                                                  2016. 03. 23. Ver 1.2.
%                                                  2016. 03. 24. Ver 1.3.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%% Preparation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%
% Clearing
%%%%%%%%%%
clear; clc; close all;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Path (including all subfolders)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath(['/Users/KimWiback/Google_Drive/Git/', ...
    'EMCS_Neural_Network/ANN_TIMIT_MATLAB']))
cd(['/Users/KimWiback/Google_Drive/Git/', ...
    'EMCS_Neural_Network/ANN_TIMIT_MATLAB'])



%%%%%%
% Data
%%%%%%
organize





%% Neural network structure verifications %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%
% Layers
%%%%%%%%
% Pattern recognition network with 256 nodes in each layer
net = patternnet([256, 256, 256, 256, 256]);



%%%%%%%%%%%
% Functions
%%%%%%%%%%%

%%% Killing mapminmax & removeconstantrows
% mapminmax squashes everything into [1, 1].
% removeconstantrows removes row vectors with unit (so constant) value.
net.input.processFcns = {};
net.output.processFcns = {};

%%% Setting transfer functions
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'logsig';
net.layers{3}.transferFcn = 'logsig';
net.layers{4}.transferFcn = 'logsig';
net.layers{5}.transferFcn = 'logsig';
% Softmax returns probabilties (sum 1) for each output.
net.layers{6}.transferFcn = 'softmax'; 

%%% Setting other functions
% Training * Validation * Test sets will be extracted in a random way.
net.divideFcn = 'dividerand';
% The extraction will be done by each sample, rather than batch.
net.divideMode = 'sample';
% Perfomance function, this is convention (has to be studied).
net.performFcn = 'crossentropy';



%%%%%%%%%%%%%%%
% Ratio control
%%%%%%%%%%%%%%%
% The training set 80%
net.divideParam.trainRatio = 80 / 100;
% The validation set 10%
net.divideParam.valRatio = 10 / 100;
% The test set 10%
net.divideParam.testRatio = 10 / 100;
fprintf(['Neural network''s structure has been established. ', ...
    'Ready for training.\n'])
%view(net)





%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
target = outputs;
% Training takes too much time, thus load already trained net.
%[net, train_record] = train(net, inputs, target);
%save('trained_net.mat', 'net')
load('trained_net.mat')



%%%%%%%%%%%%%%%%%%%%
% After the training
%%%%%%%%%%%%%%%%%%%%
prediction = net(inputs);
% Element-wise cell subtraction: delta (error) == target - prediction
error_cell = gsubtract(target, prediction);
% Getting perfromance of the net
performance = perform(net, target, prediction);
% Finding indices of the 1s (omitting number of rows)
[target_indices, ~] = vec2ind(target);
[prediction_indices, ~] = vec2ind(prediction);
% Error calculation
error_percent = sum(target_indices ~= prediction_indices) / ...
    numel(target_indices);

%%% Printing
fprintf('%s: %s\n%s: %s\n', ...
    'Performance', performance, 'Error percent (%)', error_percent)

%%% Plotting
plotroc(target, prediction)
plotconfusion(target, prediction)





%% Activations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ANN is ITERATIONS of below.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Let)
%      1. ANN with 5 hidden layers (256 nodes each) has a SINGLE input.
%      2. Input dimension is 792 * 1, so the first layer is 792 * 1 matrix.
%      3. Ouput dimension is 41 * 1, so the final layer is 41 * 1 matrix.
%      4. Activation of certain layer is sigmoid(Weight * input + bias).

%%% Demo)
% 256 * 1 == sigmoid(256 * 792 * 792 * 1 + net.b{1])
% 256 * 1 ==  sigmoid(256 * 256 * 256 * 1 + net.b{2})
% 256 * 1 ==  sigmoid(256 * 256 * 256 * 1 + net.b{3})
% 256 * 1 ==  sigmoid(256 * 256 * 256 * 1 + net.b{4})
% 256 * 1 ==  sigmoid(256 * 256 * 256 * 1 + net.b{5})
% 41 * 1 ==  sigmoid(41 * 256 * 256 * 1 + net.b{6})



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Averaging activations of each phone
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matrix dimension: the number of phones * the number of nodes
sum_hidden_1 = zeros(size(outputs, 1), 256);
sum_hidden_2 = zeros(size(outputs, 1), 256);
sum_hidden_3 = zeros(size(outputs, 1), 256);
sum_hidden_4 = zeros(size(outputs, 1), 256);
sum_hidden_5 = zeros(size(outputs, 1), 256);
averaged_sum_hidden_1 = zeros(size(outputs, 1), 256);
averaged_sum_hidden_2 = zeros(size(outputs, 1), 256);
averaged_sum_hidden_3 = zeros(size(outputs, 1), 256);
averaged_sum_hidden_4 = zeros(size(outputs, 1), 256);
averaged_sum_hidden_5 = zeros(size(outputs, 1), 256);
% Dummy indices for below loop
sum_index = 1;
print_index = 1;
number_of_matches = 0;



%%%%%%%%%%%%%%%%%%%%
% The averaging loop 
%%%%%%%%%%%%%%%%%%%%
% Phone-specific, layer-wise summation -> The averaging
for n = 1:size(outputs, 1)
    % Restoring the phone identities for comparision
    phone_id_restore = zeros(41, 1);
    phone_id_restore(phone_id_indices(n)) = 1;
    
    %%% Summate all the activations of the each phone.
    for m = 1:size(outputs, 2)
        % If the restored phone and the outputs are spot on match, proceed.
        if ~any(gsubtract(outputs(:, m), phone_id_restore))
            % Count how many matches are there.
            number_of_matches = number_of_matches + 1;
            % ANN
            at_first_hidden = net.IW{1} * inputs(:, m) + net.b{1};
            first_activation = sigmoid(at_first_hidden);
            at_second_hidden = net.LW{2, 1} * first_activation + net.b{2};
            second_activation = sigmoid(at_second_hidden);
            at_third_hidden = net.LW{3, 2} * second_activation + net.b{3};
            third_activation = sigmoid(at_third_hidden);
            at_forth_hidden = net.LW{4, 3} * third_activation + net.b{4};
            forth_activation = sigmoid(at_forth_hidden);
            at_fifth_hidden = net.LW{5, 4} * forth_activation + net.b{5};
            fifth_activation = sigmoid(at_fifth_hidden);
            % Sum them up.
            sum_hidden_1(sum_index, :) = ...
                sum_hidden_1(sum_index, :) + first_activation';
            sum_hidden_2(sum_index, :) = ...
                sum_hidden_2(sum_index, :) + second_activation';
            sum_hidden_3(sum_index, :) = ...
                sum_hidden_3(sum_index, :) + third_activation';
            sum_hidden_4(sum_index, :) = ...
                sum_hidden_4(sum_index, :) + forth_activation';
            sum_hidden_5(sum_index, :) = ...
                sum_hidden_5(sum_index, :) + fifth_activation';
        end
        
        %%%% The average
        % Do not enter the averaging at the very first loop.
        if n > 1
            % When the loop is renewd, proceed.
            if m == 1
                % Using number_of_matches
                averaged_sum_hidden_1(sum_index, :) = ...
                    sum_hidden_1(sum_index, :) / ...
                    number_of_matches;
                averaged_sum_hidden_2(sum_index, :) = ...
                    sum_hidden_2(sum_index, :) / ...
                    number_of_matches;
                averaged_sum_hidden_3(sum_index, :) = ...
                    sum_hidden_3(sum_index, :) / ...
                    number_of_matches;
                averaged_sum_hidden_4(sum_index, :) = ...
                    sum_hidden_4(sum_index, :) / ...
                    number_of_matches;
                averaged_sum_hidden_5(sum_index, :) = ...
                    sum_hidden_5(sum_index, :) / ...
                    number_of_matches;
                % Renewing the indices for the next phone
                sum_index = sum_index + 1;
                number_of_matches = 0;
            end
        end
    end
    
    %%% Printing
    if n == print_index
        print_index = print_index + 1;
        fprintf('Averaging activations over each phone: (%d/%d)\n', ...
            n, size(outputs, 1));
        % Print when the loop hits very last sample.
        if n == size(outputs, 1) && m == size(outputs, 2)
            fprintf('Done!\n');
        end
    end
end





%% Data transitions (row-wise) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%
% Normalization
%%%%%%%%%%%%%%%
% Self
normal_averaged_sum_hidden_1 = zeros(size(outputs, 1), 256);
for row = 1:size(averaged_sum_hidden_1, 1)
    normal_averaged_sum_hidden_1(row, :) = ...
        (averaged_sum_hidden_1(row, :) - ...
        min(averaged_sum_hidden_1(row, :))) ./ ...
        (max(averaged_sum_hidden_1(row, :)) - ...
        min(averaged_sum_hidden_1(row, :)));
end
% Auto (not exactly same as the auto)
normal_averaged_sum_hidden_2 = normr(averaged_sum_hidden_2);
normal_averaged_sum_hidden_3 = normr(averaged_sum_hidden_3);
normal_averaged_sum_hidden_4 = normr(averaged_sum_hidden_4);
normal_averaged_sum_hidden_5 = normr(averaged_sum_hidden_5);



%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Standardization == z-score
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
standard_averaged_sum_hidden_1 = zscore(averaged_sum_hidden_1, 1, 2);
standard_averaged_sum_hidden_2 = zscore(averaged_sum_hidden_2, 1, 2);
standard_averaged_sum_hidden_3 = zscore(averaged_sum_hidden_3, 1, 2);
standard_averaged_sum_hidden_4 = zscore(averaged_sum_hidden_4, 1, 2);
standard_averaged_sum_hidden_5 = zscore(averaged_sum_hidden_5, 1, 2);





%% Clustering (column-wise standardization) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Excluding the silence h# which is consisted of the h#, epi, and pau
clustergram(averaged_sum_hidden_1(1:40, :), 'RowLabels', ...
    phones_list_without_epi_pau(1:40), 'Standardize', 1, ...
    'colormap', 'redgreencmap', 'DisplayRange', 3);
clustergram(averaged_sum_hidden_2(1:40, :), 'RowLabels', ...
    phones_list_without_epi_pau(1:40), 'Standardize', 1, ...
    'colormap', 'redbluecmap', 'DisplayRange', 3);
clustergram(averaged_sum_hidden_3(1:40, :), 'RowLabels', ...
    phones_list_without_epi_pau(1:40), 'Standardize', 1, ...
    'colormap', colormap(hsv), 'DisplayRange', 3);
clustergram(averaged_sum_hidden_4(1:40, :), 'RowLabels', ...
    phones_list_without_epi_pau(1:40), 'Standardize', 1, ...
    'colormap', colormap(jet), 'DisplayRange', 3);
clustergram(averaged_sum_hidden_5(1:40, :), 'RowLabels', ...
    phones_list_without_epi_pau(1:40), 'Standardize', 1, ...
    'colormap', flipud(colormap(gray)), 'DisplayRange', 3);