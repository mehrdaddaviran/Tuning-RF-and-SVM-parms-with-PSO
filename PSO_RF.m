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
clc
clear
close all
profile on
%% Load your dataset and split into training and testing sets
% load Data
Data();
X_train = Geospatialdata(:,1:end-1);
Y_train = Geospatialdata(:,end);
DataNum = size(Geospatialdata,1);
%%
disp('PSO-RF');
disp('===========================================');
TrPercentinput = input('Enter the Percent of Train data:   ');
%%disp('Define the PSO options');
disp('===========================================');
maxIteration = input('Enter the Max Iterations:   ');
swarmSize = input('Enter the SwarmSize:   ');
disp('===========================================');
%% Test and Train Data
TrPercent = TrPercentinput;
TrNum = round(DataNum * TrPercent / 100);
TsNum = DataNum - TrNum;

R = randperm(DataNum);
trIndex = R(1 : TrNum);
tsIndex = R(1+TrNum : end);

Xtr = X_train(trIndex,:);
Ytr = Y_train(trIndex,:);

Xts = X_train(tsIndex,:);
Yts = Y_train(tsIndex,:);
% K Fold Cross Validation Indices
K_fold = 5;

%% Define the fitness function that evaluates the random forest model
fitness_function = @(params)fit_random_forest(params, Xtr, Ytr, K_fold);

%% Define the PSO options
options = optimoptions('particleswarm', 'MaxIterations', maxIteration, 'SwarmSize', swarmSize, 'Display', 'iter');

%% Define the parameter bounds and initial swarm
num_params = 3; % number of random forest parameters to optimize
lb = [1 1 1]; % lower bounds for parameters
ub = [100 100 100]; % upper bounds for parameters
initial_swarm = rand(options.SwarmSize, num_params).*(ub-lb) + lb;

%% Run the PSO algorithm
[best_params, fval] = particleswarm(fitness_function, num_params, lb, ub, options);

%% Train the random forest model with the best parameters
num_trees = round(best_params(1));
max_depth = best_params(2);
min_leaf_size = best_params(3);
rf_modelFinal = TreeBagger(num_trees, Xtr, Ytr, 'Method', 'classification', 'MaxNumSplits', max_depth, 'MinLeafSize', min_leaf_size);

%% Evaluate the trained model on the testing set
[Yhard, Ysoft] = predict(rf_modelFinal, Xts);
Yhardpred2 = str2double(Yhard);
mse = mean((Yts - (round(Yhardpred2))).^2);
% Evalute Test Data
Out.mseTesteingset=mse;
BTest = confusionmat(Yts,Yhardpred2);
Out.performanceEvaluteTestData = 100*sum(diag(BTest))/sum(BTest(:));
S = 0;
for ii = 1:size(BTest,1)
    S = S + BTest(ii,ii);
end
per = S/numel(Yhardpred2);
Out.TestPerformance = 100*per;
%figure,plotconfusion(TestData.Targets',Class')
figure('Name','Confusion Plot for TestData'), plotconfusion(Yts',Yhardpred2');
%% Evalute All Data
[YhardAll, YsoftAll] = predict(rf_modelFinal, X_train);
Yhardpred3 = str2double(YhardAll);
ClassAll=Yhardpred3;
B_ALL = confusionmat(Y_train,ClassAll);
Out.performanceEvaluteAllData = 100*sum(diag(B_ALL))/sum(B_ALL(:));
S = 0;
for ii = 1:size(B_ALL,1)
    S = S + B_ALL(ii,ii);
end
per = S/numel(ClassAll);
Out.AllPerformance = 100*per;

TP = B_ALL(1,1);
TN = B_ALL(2,2);
FP = B_ALL(1,2);
FN = B_ALL(2,1);

Out.Sensivity = (TP/(TP   + TN));
Out.Specificity = (TN/(FP + TN));
Out.Perecision = (TP/(TP  + FP));
Out.Accuracy = (TP + TN)/(TP + TN + FP + FN);

%
figure('Name','Confusion Plot for All Data'), plotconfusion(Y_train',ClassAll')
disp(Out)
%%
% Final Model Based on RF tuned parms
filenameF = 'input data.xlsx';
CCC = xlsread(filenameF,3); 
[Y_Final,scores]= predict(rf_modelFinal, CCC);
profile viewer
%%
disp('PSO-RF Results');
disp('===========================================');
disp('Number of trees:' );
disp(num_trees);
disp('===========================================');
disp('Number of splits:' );
disp(min_leaf_size);
disp('===========================================');
disp('Depth:' );
disp(max_depth);
disp('===========================================');
disp('Y_Final,scores are the final PSO-RF based results');
disp('===========================================');
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