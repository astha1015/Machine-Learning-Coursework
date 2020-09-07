clear; %Clear variables and functions from memoryclc;%Clear command window% input for ANN3 L = [4 4 4 3 3];% Initializing the required parameterstotalfolds = 10;min_test_errors = []; % array or Min test errors min_train_errors = []; % array for Min training errors min_avg_test_error = 0;  % Min average test errormin_avg_train_error = 0;	% Min average train errormin_error_all_run = Inf; % Min error of all runavg_classification_accuracy_train = 0;avg_classification_accuracy_test = 0;avg_train_accuracy=[];avg_test_accuracy=[];%%%%%%%%%% ANN starts here %%%%%%%%%%%%%%%%%for loop = 1:totalfolds        alpha = 0.2;       target_mse = 0.05; % one of the exit conditions    Max_Epoch = 2000; % another exit condition    Min_Error = Inf; % Stores minimum train error    Min_Error_Test = Inf;    Min_Error_Epoch = -1; % Iteration that gives minimum error     epoch = 0;          train_mse = Inf; % initialization of training error     test_mse = Inf;  % initialization of test error    TrainErr = [];  % array for storing training error    TestErr = [];  % array for storing test error    Accuracy_Train=[];    Accuracy_Test =[];    Epo = [];     % array for storing the iteration sequence    #gamma = 0.5;       % Loading dataset for training purpose         FileName   = ['folds/train',num2str(loop),'.txt'];    train = load(FileName);    [train_r, train_c] = size(train);    ip_mat = train(:,1:train_c-1);    ip_mat;    [Nx, P] = size(ip_mat); % Nx = number of input rows, P = number of columns (features)    tmp_o_mat = train(:,train_c);       o_mat = zeros(Nx, 3);    % Create matrix classes     for i = 1:Nx         for j = 1:3            if tmp_o_mat(i) == j                o_mat(i,j) = 1;            else                o_mat(i,j) = 0;            end        end    end    [Ny,K] = size(o_mat);  %Ny = number of target output, K= number of output class          % Loading Dataset for Testing purpose           FileName   = ['folds/test',num2str(loop),'.txt'];    test=load(FileName);    [test_r, test_c] = size(test);       test_i_mat = test(:,1:test_c-1);    [test_R, test_C] = size(test_i_mat);    % Creating a temporary matrix     tmp_test_o_mat = test(:,test_c);    test_o_mat = zeros(test_R, 3);     % Create operational matrix    for i = 1:test_R        for j = 1:3            if tmp_test_o_mat(i) == j              test_o_mat(i,j) = 1;            else              test_o_mat(i,j) = 0;            end        end        end    [test_o_r, test_o_c] = size(test_o_mat);    %throw error if output row is not equal to input row         if Nx ~= Ny           error ('The input/output sample sizes do not match');    end    % throw error if number of nodes in input layer is not equal to number of features    if L(1) ~= P          error ('The number of input nodes must be equal to the size of the features')'     end     if L(end) ~= K          error ('The number of output node should be equal to K')'     end         % Initialize Weight or Beta Matrix of the neural network    B=cell(length(L)-1,1);     for i=1:length(L)-1       B{i} =(1.4.*rand(L(i)+1,L(i+1))-0.7);	% Assign uniform random values in [-0.7, 0.7]    end    % store Best Beta (BB) in a matrix and then write in a file at the end of program    BB=cell(length(L)-1,1);    for i=1:length(L)-1      BB{i} =[rand(L(i)+1,L(i+1))];	% assigning Random values to the elements    end    % Intializing matrix for the term T     T=cell(length(L),1);    for i=1:length(L)        T{i} =ones (L(i),1);    end    % Initializing matrix for the term Z    Z=cell(length(L),1);    for i=1:length(L)-1        Z{i} =zeros (L(i)+1,1);     end    Z{end} =zeros (L(end),1);     % Initializing matrix for the term Velocity     V=cell(length(L),1);    for i=1:length(L)-1        V{i}=(0.*rand(L(i)+1,L(i+1)));    end    % Intializing matrix for the term T (here, T is replaced by U)    U=cell(length(L),1);    for i=1:length(L)           U{i} =ones (L(i),1);    end    % Intializing matrix for term X for test    X=cell(length(L),1);    for i=1:length(L)-1        X{i} =zeros (L(i)+1,1);     end    X{end} =zeros (L(end),1);      % Initializing matrix for term delta (here, d) for error    d=cell(length(L),1);    for i=1:length(L)        d{i} =zeros (L(i),1);    end    % Store average error    avg_error=cell(length(L)-1,1);     for i = 1:length(L)-1        avg_error{i} = cell(2,1);    end    % Store train and test data average error    for i = 1:length(L)-1        avg_error{i}{1} = zeros(L(i+1),L(i));         avg_error{i}{2} = zeros(L(i+1),1);    end       fprintf('calculating min test error and corresponding epoch for loop:%s \n',num2str(loop));       % Get layers error value on progations     Z{1}=[ip_mat ones(Nx, 1)]';     Y=o_mat';    while (train_mse > target_mse) && (epoch < Max_Epoch)   % outer loop with exit conditions         CSqErr=0;         accuracy_train = 0;           % start of forward propagation for training                   for i=1:length(L)-1              T{i+1} = B{i}' * Z{i};              if (i+1)<length(L)                Z{i+1}=[(1./(1+exp(-T{i+1}))); ones(Nx,1)'];              else                  Z{i+1}=(1./(1+exp(-T{i+1})));               end          end                  CSqErr= CSqErr+sum(sum(((Y-Z{end}).^2),1));               CSqErr = CSqErr/L(end);  % Normalizing the Error based on the number of output Nodes                      % Compute error term delta 'd' for each of the node except the input unit        d{end}=(Z{end}-Y).*Z{end}.*(1-Z{end});         for i=length(L)-1:-1:2              W=Z{i}(1:end-1,:).*(1-Z{i}(1:end-1,:)); D= d{i+1}';              for m = 1:Nx                                               d{i}(:,m)=W(:,m).*sum((D(m,:).*B{i}(1:end-1,:)),2);                end        end          %  Updating the parameters/weights        for i=1:length(L)-1           W = Z{i}(1:end-1,:);          V1 = zeros(L(i),L(i+1));          V2 = zeros(1,L(i+1));          D = d{i+1}';          for m = 1:Nx              V1 = V1 + (W(:,m)*D(m,:));              V2 = V2 + D(m,:);               end                 B{i}(1:end-1,:)=B{i}(1:end-1,:)-(alpha/Nx).*V1;                       B{i}(end,:) = B{i}(end,:)-(alpha/Nx).*V2;  			        end         %Saving the node of the neural network and corresponding structure        L;    % saving the predicted weight B with least error        for i=1:min(size(B))            B{i};        end                  % Start of forward propagation for Testing         CSqErr_test=0;        accuracy_test = 0;        for j=1:test_R                   		          X{1} = [test_i_mat(j,:) 1]';            Yk   = test_o_mat(j,:)';           for i=1:length(L)-1            U{i+1} = B{i}' * X{i};            if (i+1)<length(L)              X{i+1}=[(1./(1+exp(-U{i+1}))) ;1];            else                X{i+1}=(1./(1+exp(-U{i+1})));             end          end  % End                     %Test accuracy          [max_value_x_test, x_class_test]= max(X{end});          [max_value_y_test, y_class_test] = max(Yk);          if x_class_test == y_class_test            accuracy_test = accuracy_test + 1;          end                                      CSqErr_test= CSqErr_test+sum(sum(((Yk-X{end}).^2),1));        end                         CSqErr_test = ((CSqErr_test) /(test_R)); % Average test error             test_mse = CSqErr_test;                 TestErr = [TestErr test_mse];                               if test_mse < Min_Error_Test          Min_Error_Test = test_mse;          Min_Error_Epoch=epoch;        end                acc_row=1;        acc_col=3;        for m=1:Nx          [max_value_x_train, x_class_train]= max(Z{end}(acc_row:acc_col));          [max_value_y_train, y_class_train] = max(Y(acc_row:acc_col));          if x_class_train == y_class_train            accuracy_train = accuracy_train + 1;          end          acc_row = acc_row + 3;          acc_col = acc_col + 3;         end                  CSqErr = (CSqErr) /(Nx);        % Average error of N sample after an epoch         train_mse = CSqErr;        TrainErr = [TrainErr train_mse];        acc_test = (accuracy_test/15) * 100;        acc_train = (accuracy_train/135) * 100;        Accuracy_Test = [Accuracy_Test acc_test];        Accuracy_Train = [Accuracy_Train acc_train];                epoch  = epoch+1;        Epo = [Epo epoch];                if train_mse < Min_Error          Min_Error = train_mse;                    for i=1:length(L)-1              BB{i} = B{i};                  end        end            end    % end of while loop      Min_Error_Test    %print minimum error on console    Min_Error_Epoch   %print minimum error iteration on consol            min_train_errors = [min_train_errors Min_Error];    min_test_errors = [min_test_errors Min_Error_Test];    avg_train_accuracy  = [avg_train_accuracy sum(Accuracy_Train)/length(Accuracy_Train)];    avg_test_accuracy = [avg_test_accuracy sum(Accuracy_Test)/length(Accuracy_Test)];    % in a file print the best beta  value     dlmwrite('best_beta_value_ANN3.txt', L,'-append');    for i = 1:length(L)-1      dlmwrite('best_beta_value_ANN3.txt', BB{i}, '-append', 'roffset', 2);     end    % finding the best beta value among all available    if(Min_Error < min_error_all_run)        Best_all_BB = BB;        min_error_all_run = Min_Error;        Best_all_TrainErr = TrainErr;        Best_all_TestErr = TestErr;    endend% Endmin_train_errors    % print min_train_errors vectormin_test_errors     % print min_test_errors vectoravg_train_accuracy  %print average training accuracyavg_test_accuracy   %print average test accuracymin_avg_train_error=sum(min_train_errors)/totalfolds;  %calcuate average for min training errormin_avg_train_error   %print min_avg_train_errormin_avg_test_error=sum(min_test_errors)/totalfolds;    %calculate average for min test errormin_avg_test_error    %print min_avg_test_erroravg_classification_accuracy_train = sum(avg_train_accuracy)/totalfolds %print avg_classification_accuracy_trainavg_classification_accuracy_test = sum(avg_test_accuracy)/totalfolds %print avg_classification_accuracy_test%write best beta form all runs to the filedlmwrite('best_beta_value_ANN3.txt', L, '-append');for i = 1:length(L)-1   dlmwrite('best_beta_value_ANN3.txt', Best_all_BB{i}, '-append', 'roffset', 2); end% Plot iteration versus average MSE graph for test and train figure()               plot (Epo(1:2000),Best_all_TrainErr(1:2000))  holdplot (Epo(1:2000),Best_all_TestErr(1:2000)) title('Train/Test Error on each Iterations')xlabel('Iteration')ylabel('Averge Error') legend('Train Error', 'Test Error')%Plot iteration versus average classification accuracy graph figure()plot (Epo(1:2000),Accuracy_Train(1:2000))  holdplot (Epo(1:2000),Accuracy_Test(1:2000)) title('Train/Test Accuracy on each Iterations')xlabel('Iteration')ylabel('Averge Accuracy') legend('Train Accuracy', 'Test Accuracy') 