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
function CrossValIndices = CrossValKfold(Targets,K)
% Set Params for K-fold
N = numel(Targets);
CrossValIndices = CrossValInd(N,K);
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