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
function Ind = CrossValInd(N,K)
Ind = zeros(N,1);
h = 1:N;
% compute fold id's for every observation in the  group
q = ceil(K*(1:N)/N);
% and permute them to try to balance among all groups
pq = randperm(K);
% randomly assign the id's to the observations of this group
randInd = randperm(N);
Ind(h(randInd))=pq(q);

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