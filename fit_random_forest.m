%%________________________________________________________________________________
%  Hyperparameter Tuning of Random Forest and SVM with Particle swarm optimization
%
%  Developed in MATLAB R2021b
%
%  Authors and programmer*: 
%
%  1*.Mehrdad Daviran
%   "Department of Mining Engineering, Amirkabir University of Technology, Tehran, Iran"
%        e-Mail: mehrdaddaviran@yahoo.com
%                mehrdaddaviran@aut.ac.ir
%  2.Abbas Maghsoudi
%   "Department of Mining Engineering, Amirkabir University of Technology, Tehran, Iran"
%        e-Mail: a.maghsoudi@aut.ac.ir
%              
%  3.Reza Ghezelbash
%   "School of Mining Engineering, College of Engineering, University of Tehran, Tehran, Iran"
%        e-Mail: r.ghezelbash@ut.ac.ir
%
%
%   Main paper:
% Optimized AI-MPM: Application of PSO for tuning the hyperparameters of SVM and RF algorithms
%%
function [mse, z] = fit_random_forest(params, X_train, Y_train, K_fold)
global NFE 
if isempty(NFE)
    NFE = 0;
end
NFE = NFE + 1;
CrossValIndices = CrossValKfold(Y_train,K_fold);
K = max(CrossValIndices);
Features = X_train;
Label = Y_train;
per = zeros(1,K);
for i = 1:K
    
    training = Features(CrossValIndices~=i,:);
    group = Label(CrossValIndices~=i);
    group=group';
    valid = Features(CrossValIndices==i,:);
    label = Label(CrossValIndices==i);
        % Train a random forest model with the specified parameters
    num_trees = round(params(1));
    max_depth = round(params(2));
    min_leaf_size = params(3);
    rf_model = TreeBagger(num_trees, training, group, 'Method', 'classification', 'MaxNumSplits', max_depth, 'MinLeafSize', min_leaf_size);
    
    % Predict on the testing set and calculate the mean squared error (MSE)
   [Yhardpred, Ysoftpred] = predict(rf_model, valid);
   Yhardpred1 = str2double(Yhardpred);
    mse = mean((label - round(Yhardpred1)).^2);
    Class=Yhardpred1';
    B = confusionmat(label,Class);
    S = 0;
    for ii = 1:size(B,1)
        S = S + B(ii,ii);
    end
    per(i) = S/numel(label);
end
z = 1 - mean(per);
end
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        Mehrdad Daviran, Abbas Maghsoudi, Reza Ghezelbash             %
%                                                                      %             
%               Amirkabir University of Technology                     %
%                      University of Tehran                            %
%                                                                      %
%                                                                      %      
%              Hyperparameter Tuning of Random Forest                  %
%                 with Particle swarm optimization                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%