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
X = Geospatialdata(:,1:end-1);
Y = Geospatialdata(:,end);
DataNum = size(Geospatialdata,1);
%%
disp('PSO-SVM');
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

Xtr = X(trIndex,:);
Ytr = Y(trIndex,:);

Xts = X(tsIndex,:);
Yts = Y(tsIndex,:);
%%
% Set the parameters
max_iter = maxIteration;
pop_size = swarmSize;
c1 = 2;
c2 = 2;
w = 0.9;
lb = 0;
ub = 1;
K_fold = 5;
%%
% Initialize the population
pop1 = lb + (ub-lb)*rand(pop_size,2);
pop2 = (pop1(:,1))*5 + 0.001;
pop3 = (pop1(:,2))*10 + 0.001;
pop = [pop2 pop3];
% Initialize the best particle and its fitness
%[gbest_fitness, gbest_index] = inf;
% Initialize the personal best position and fitness
% Initialize the best particle and its fitness
gbest_fitness = inf;
gbest_position = pop(1,:);

pbest_position = pop;
pbest_fitness = inf(1,pop_size);
vel = zeros(pop_size,2);
% Main loop
for iter=1:max_iter
    % Evaluate the fitness of each particle
    for i=1:pop_size
        % Compute the SVM fitness
        fitness(i) = svm(Xtr,Ytr,pop(i,1),pop(i,2),K_fold);
        
        % Update the personal best position and fitness
        if fitness(i) < pbest_fitness(i)
            pbest_fitness(i) = fitness(i);
            pbest_position(i,:) = pop(i,:);
        end
        
        % Update the global best position and fitness
        if fitness(i) < gbest_fitness
            gbest_fitness = fitness(i);
            gbest_position = pop(i,:);
        end
    end
  
    % Update the velocity and position of each particle
   for i=1:pop_size
        % Update the velocity
      vel(i,:) = w*vel(i,:) + c1*rand()*(pbest_position(i,:)-pop(i,:)) + c2*rand()*(gbest_position-pop(i,:));
        
        % Check the velocity limits
        vel(i,:) = min(vel(i,:),ub);
       vel(i,:) = max(vel(i,:),lb);
        
        % Update the position
       pop(i,:) = pop(i,:) + vel(i,:);
        
        % Check the position limits
        pop(i,:) = min(pop(i,:),ub);
        pop(i,:) = max(pop(i,:),lb);
   end
    
    % Update the SVM parameters
   C = gbest_position(1);
   gamma = gbest_position(2);
    
    svm_model = fitcsvm(Xtr,Ytr,'KernelFunction','rbf','BoxConstraint',C,'KernelScale',gamma,'Standardize',true);
    Trained_model = svm_model;
end
%% Evaluate the trained model on the testing set
[Yhard, Ysoft] = predict(Trained_model, Xts);
Yhardpred2 = Yhard;
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
[YhardAll, YsoftAll] = predict(Trained_model, X);
Yhardpred3 = YhardAll;
ClassAll=Yhardpred3;
B_ALL = confusionmat(Y,ClassAll);
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
figure('Name','Confusion Plot for All Data'), plotconfusion(Y',ClassAll')
disp(Out)
%%
% Final Model Based on RF tuned parms
filenameF = 'input data.xlsx';
CCC = xlsread(filenameF,3); 
[Y_Final,scores]= predict(Trained_model, CCC);
profile viewer
%%
disp('PSO-SVM Results');
disp('===========================================');
disp('BoxConstraint:' );
disp(C);
disp('===========================================');
disp('KernelScale:' );
disp(gamma);
disp('===========================================');
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
