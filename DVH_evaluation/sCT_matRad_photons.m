clear
%% Example: Photon Treatment Plan
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright 2017 the matRad development team.
%
% This file is part of the matRad project. It is subject to the license
% terms in the LICENSE file found in the top-level directory of this
% distribution and at https://github.com/e0404/matRad/LICENSES.txt. No part
% of the matRad project, including this file, may be copied, modified,
% propagated, or distributed except according to the terms contained in the
% LICENSE file.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
% In this example we will show
% (i) how to load patient data into matRad
% (ii) how to setup a photon dose calculation and
% (iii) how to inversely optimize beamlet intensities
% (iv) how to visually and quantitatively evaluate the result

%% Patient Data Import
% Let's begin with a clear Matlab environment. Then, import the TG119
% phantom into your workspace. The phantom is comprised of a 'ct' and 'cst'
% structure defining the CT images and the structure set. Make sure the
% matRad root directory with all its subdirectories is added to the Matlab
% search path.

matRad_rc; %If this throws an error, run it from the parent directory first to set the paths

data_folder="<PATH>/dose_calcs/pix2pix";
NN_models=["MODEL_LIST"];

for NN_model =NN_models
    

    NN_results_folder=fullfile(data_folder,NN_model);
    all_dvh_params_file_name= 'all_dvh_params'+NN_model+'.csv';
    all_dvh_diffs_file_name='all_dvh_diffs'+NN_model+'.csv';
    all_dvh_params_file_path=fullfile(data_folder,all_dvh_params_file_name);
    all_dvh_diffs_file_path=fullfile(data_folder,all_dvh_diffs_file_name);
    if isfile(all_dvh_params_file_path) || isfile(all_dvh_diffs_file_path)
        delete(all_dvh_params_file_path);
        delete(all_dvh_diffs_file_path);
    end
    
    
    subdir="real_overwritten_matrad";%always firstly compute for real and then pass it to sCT
    D = dir(fullfile(NN_results_folder,subdir)); % A is a struct ... first elements are '.' and '..' used for navigation.
    for k = 3:length(D) % avoid using the first ones
        patient = D(k).name;
		subdir="real_overwritten_matrad";%always firstly compute for real and then pass it to sCT
    
        patient_matrad_CT_file = fullfile(NN_results_folder,subdir, patient);
        load(patient_matrad_CT_file);
        matRadGUI
    
        %subdirs=["fake","real"];
        %for subdir = subdirs
        %     D = dir(fullfile(NN_results_folder,subdir)); % A is a struct ... first elements are '.' and '..' used for navigation.
        %     for k = 3:length(D) % avoid using the first ones
        %         currD = D(k).name;
        %         files=[];
        %         patient=currD;
        %         dir_path_field = fullfile(NN_results_folder,subdir, patient);
        %         cd(fullfile(NN_results_folder,subdir,currD))
        %         [fileList, patient_listbox] = matRad_scanDicomImportFolder(dir_path_field);
    
    
        %%
        % The file TG119.mat contains two Matlab variables. Let's check what we
        % have just imported. First, the 'ct' variable comprises the ct cube along
        %with some meta information describing properties of the ct cube (cube
        % dimensions, resolution, number of CT scenarios). Please note that
        %multiple ct cubes (e.g. 4D CT) can be stored in the cell array ct.cube{}
        %display(ct);
    
        %%
        % The 'cst' cell array defines volumes of interests along with information
        % required for optimization. Each row belongs to one certain volume of
        % interest (VOI), whereas each column defines different properties.
        % Specifically, the second and third column  show the name and the type of
        % the structure. The type can be set to OAR, TARGET or IGNORED. The fourth
        % column contains a linear index vector that lists all voxels belonging to
        % a certain VOI.
        %display(cst);
    
        %%
        % The fifth column represents meta parameters for optimization. The overlap
        % priority is used to resolve ambiguities of overlapping structures (voxels
        % belonging to multiple structures will only be assigned to the VOI(s) with
        % the highest overlap priority, i.e.. the lowest value). The parameters
        % alphaX and betaX correspond to the tissue's photon-radiosensitivity
        % parameter of the linear quadratic model. These parameter are required for
        % biological treatment planning using a variable RBE. Let's output the meta
        % optimization parameter of the target, which is stored in the thrid row:
        %ixTarget = 3;
        %display(cst{ixTarget,5});
    
        %%
        % The sixth column contains optimization information such as objectives and
        % constraints which are required to calculate the objective function value.
        % Please note, that multiple objectives/constraints can be defined for
        % individual structures. Here, we have defined a squared deviation
        % objective making it 'expensive/costly' for the optimizer to over- and
        % underdose the target structure (both are equally important).
        %display(cst{ixTarget,6});
    
        %% Treatment Plan
        % The next step is to define your treatment plan labeled as 'pln'. This
        % matlab structure requires input from the treatment planner and defines
        % the most important cornerstones of your treatment plan.
    
        %%
        % First of all, we need to define what kind of radiation modality we would
        % like to use. Possible values are photons, protons or carbon. In this case
        % we want to use photons. Then, we need to define a treatment machine to
        % correctly load the corresponding base data. matRad includes base data for
        % generic photon linear accelerator called 'Generic'. By this means matRad
        % will look for 'photons_Generic.mat' in our root directory and will use
        % the data provided in there for dose calculation
        pln.radiationMode = 'photons';
        pln.machine       = 'Generic';
    
        %%
        % Define the flavor of optimization along with the quantity that should be
        % used for optimization. Possible values are (none: physical optimization;
        % const_RBExD: constant RBE of 1.1; LEMIV_effect: effect-based
        % optimization; LEMIV_RBExD: optimization of RBE-weighted dose. As we are
        % using photons, we simply set the parameter to 'none' thereby indicating
        % the physical dose should be optimized.
        pln.propOpt.bioOptimization = 'none';
    
        %%
        % Now we have to set some beam parameters. We can define multiple beam
        % angles for the treatment and pass these to the plan as a vector. matRad
        % will then interpret the vector as multiple beams. In this case, we define
        % linear spaced beams from 0 degree to 359 degree in 40 degree steps. This
        % results in 9 beams. All corresponding couch angles are set to 0 at this
        % point. Moreover, we set the bixelWidth to 5, which results in a beamlet
        % size of 5 x 5 mm in the isocenter plane. The number of fractions is set
        % to 30. Internally, matRad considers the fraction dose for optimization,
        % however, objetives and constraints are defined for the entire treatment.
        pln.numOfFractions         = 1;
        pln.propStf.gantryAngles   = [0 23 50 75 95 110 150 170 190 210 250 265 285 310 337];
        pln.propStf.couchAngles    = zeros(1,numel(pln.propStf.gantryAngles));
        pln.propStf.bixelWidth     = 4;
    
        %%
        % Obtain the number of beams and voxels from the existing variables and
        % calculate the iso-center which is per default the center of gravity of
        % all target voxels.
        pln.propStf.numOfBeams      = numel(pln.propStf.gantryAngles);
        pln.propStf.isoCenter       = ones(pln.propStf.numOfBeams,1) * matRad_getIsoCenter(cst,ct,0);
    
        %% dose calculation settings
        % set resolution of dose calculation and optimization
        pln.propDoseCalc.doseGrid.resolution.x = 4; % [mm]
        pln.propDoseCalc.doseGrid.resolution.y = 4; % [mm]
        pln.propDoseCalc.doseGrid.resolution.z = 4; % [mm]
    
        %%
        % Enable sequencing and disable direct aperture optimization (DAO) for now.
        % A DAO optimization is shown in a seperate example.
        pln.propOpt.runSequencing = 1;
        pln.numOfLevels=5;
        pln.propOpt.runDAO        = 0;
    
        %%
        % and et voila our treatment plan structure is ready. Lets have a look:
        display(pln);
    
        %% Generate Beam Geometry STF
        % The steering file struct comprises the complete beam geometry along with
        % ray position, pencil beam positions and energies, source to axis distance (SAD) etc.
        stf = matRad_generateStf(ct,cst,pln);
    
        %%
        % Let's display the beam geometry information of the 6th beam
        display(stf(6));
    
        %% Dose Calculation
        % Let's generate dosimetric information by pre-computing dose influence
        % matrices for unit beamlet intensities. Having dose influences available
        % allows subsequent inverse optimization.
        dij = matRad_calcPhotonDose(ct,stf,pln,cst);
    
        %% Inverse Optimization for IMRT
        % The goal of the fluence optimization is to find a set of beamlet/pencil
        % beam weights which yield the best possible dose distribution according to
        % the clinical objectives and constraints underlying the radiation
        % treatment. Once the optimization has finished, trigger once the GUI to
        % visualize the optimized dose cubes.
        resultGUI = matRad_fluenceOptimization(dij,cst,pln);
        matRadGUI;
        weights=resultGUI.w;
    
        [dvh,qi]=matRad_indicatorWrapper(cst,pln, resultGUI);
        [filepath,name,ext] = fileparts (patient);
    
        
        dvh_table_real=struct2table(qi);
        dvh_table_real1=dvh_table_real;
        dvh_table_real=dvh_table_real(:,1:10);
        dvh_table_real.patient= repmat(char(name),length(qi),1);
        dvh_table_real.ct_type = repmat(char("real"),length(qi),1);
    
		clear resultGUI ct cst* idx qi* dvh dij;
        subdir="fake_matrad";
        patient_matrad_CT_file = fullfile(NN_results_folder,subdir, patient);
        load(patient_matrad_CT_file);
		delete(matRadGUI);
    
        % dij= matRad_calcPhotonDose(ct,stf,pln,cst);
        resultGUI = matRad_calcDoseDirect(ct,stf,pln,cst, weights);
        matRadGUI;
    
        [dvh,qi]=matRad_indicatorWrapper(cst,pln, resultGUI);
        [filepath,name,ext] = fileparts (patient);
        dvh_table_fake=struct2table(qi);
        dvh_table_fake1=dvh_table_fake;
        dvh_table_fake=dvh_table_fake(:,1:10);
        dvh_table_fake.patient= repmat(char(name),length(qi),1);
        dvh_table_fake.ct_type = repmat(char("fake"),length(qi),1);
    
        dvh_table_real = sortrows(dvh_table_real);
        dvh_table_fake = sortrows(dvh_table_fake);
        dvh_table_fake.NN_model= repmat(char(NN_model),height(dvh_table_fake),1);
        dvh_table_real.NN_model= repmat(char(NN_model),height(dvh_table_real),1);
        dvh_table_patient=[dvh_table_real;dvh_table_fake];
    
        dvh_table_diff=dvh_table_real;
        for i =1:height(dvh_table_real)
            for j=["mean","std","max", "min", "D_2","D_5", "D_50", "D_95", "D_98"]
                if cell2mat(dvh_table_real{i,"name"})==cell2mat(dvh_table_fake{i,"name"})
                    dvh_table_diff{i,j}=(dvh_table_real{i,j}-dvh_table_fake{i,j})/dvh_table_real{i,j}*100;
                else
                    dvh_table_diff{i,j}="error";
                end
            end
        end
    
    
        cst_table=cell2table(cst);
        cst_table=cst_table(:,2:3);
        cst_table = renamevars(cst_table,["cst2","cst3"],["name","objective"]);
        cst_table.objective = categorical(cst_table.objective);
    
        idx = cst_table.objective == 'OAR' | cst_table.objective == 'TARGET' ;
        cst_table = cst_table(idx,:);
    
        dvh_table_diff = innerjoin(cst_table,dvh_table_diff);
    
    
    
        if isfile(all_dvh_params_file_path) && isfile(all_dvh_diffs_file_path)
    
            dvh_table_patient_existed=readtable(all_dvh_params_file_path);
            dvh_table_patient_existed.patient=cellstr(dvh_table_patient_existed{:,"patient"});
            dvh_table_patient_existed.ct_type=cellstr(dvh_table_patient_existed{:,"ct_type"});
            dvh_table_patient_existed.NN_model=cellstr(dvh_table_patient_existed{:,"NN_model"});
            dvh_table_patient.patient=cellstr(dvh_table_patient.patient);
            dvh_table_patient.ct_type=cellstr(dvh_table_patient.ct_type);
            dvh_table_patient.NN_model=cellstr(dvh_table_patient.NN_model);
            dvh_table_patient=[dvh_table_patient;dvh_table_patient_existed];
    
            dvh_table_diff_existed=readtable(all_dvh_diffs_file_path);
            dvh_table_diff_existed.patient=cellstr(dvh_table_diff_existed{:,"patient"});
            dvh_table_diff_existed.ct_type=cellstr(dvh_table_diff_existed{:,"ct_type"});
            dvh_table_diff_existed.NN_model=cellstr(dvh_table_diff_existed{:,"NN_model"});
            dvh_table_diff.patient=cellstr(dvh_table_diff{:,"patient"});
            dvh_table_diff.ct_type=cellstr(dvh_table_diff{:,"ct_type"});
            dvh_table_diff.NN_model=cellstr(dvh_table_diff{:,"NN_model"});
            dvh_table_diff=[dvh_table_diff; dvh_table_diff_existed];
        end
    
        writetable(dvh_table_patient, all_dvh_params_file_path);
        writetable(dvh_table_diff, all_dvh_diffs_file_path);
		close all;
		clear dvh*;
		clear dij pln stf cst* idx ct qi* weights ct  resultGUI;
    
    end

end





% %% Plot the Resulting Dose Slice
% % Let's plot the transversal iso-center dose slice
% slice = round(pln.propStf.isoCenter(1,3)./ct.resolution.z);
% figure
% imagesc(resultGUI.physicalDose(:,:,slice)),colorbar, colormap(jet);
% 
% %% Now let's create another treatment plan but this time use a coarser beam spacing.
% % Instead of 40 degree spacing use a 50 degree geantry beam spacing
% pln.propStf.gantryAngles = [0:50:359];
% pln.propStf.couchAngles  = zeros(1,numel(pln.propStf.gantryAngles));
% pln.propStf.numOfBeams   = numel(pln.propStf.gantryAngles);
% stf                      = matRad_generateStf(ct,cst,pln);
% pln.propStf.isoCenter    = vertcat(stf.isoCenter);
% dij                      = matRad_calcPhotonDose(ct,stf,pln,cst);
% resultGUI_coarse         = matRad_fluenceOptimization(dij,cst,pln);
% 
% %%  Visual Comparison of results
% % Let's compare the new recalculation against the optimization result.
% % Check if you have added all subdirectories to the Matlab search path,
% % otherwise it will not find the plotting function
% plane      = 3;
% doseWindow = [0 max([resultGUI.physicalDose(:); resultGUI_coarse.physicalDose(:)])];
% 
% figure,title('original plan - fine beam spacing')
% matRad_plotSliceWrapper(gca,ct,cst,1,resultGUI.physicalDose,plane,slice,[],0.75,colorcube,[],doseWindow,[]);
% figure,title('modified plan - coarse beam spacing')
% matRad_plotSliceWrapper(gca,ct,cst,1,resultGUI_coarse.physicalDose,plane,slice,[],0.75,colorcube,[],doseWindow,[]);
% 
% %% 
% % At this point we would like to see the absolute difference of the first 
% % optimization (finer beam spacing) and the second optimization (coarser 
% % beam spacing)
% absDiffCube = resultGUI.physicalDose-resultGUI_coarse.physicalDose;
% figure,title( 'fine beam spacing plan - coarse beam spacing plan')
% matRad_plotSliceWrapper(gca,ct,cst,1,absDiffCube,plane,slice,[],[],colorcube);
% 
% %% Obtain dose statistics
% % Two more columns will be added to the cst structure depicting the DVH and
% % standard dose statistics such as D95,D98, mean dose, max dose etc.
% [dvh,qi]               = matRad_indicatorWrapper(cst,pln,resultGUI);
% [dvh_coarse,qi_coarse] = matRad_indicatorWrapper(cst,pln,resultGUI_coarse);
% 
% %%
% % The treatment plan using more beams should in principle result in a
% % better OAR sparing. Therefore lets have a look at the D95 of the OAR of 
% % both plans
% ixOAR = 2;
% display(qi(ixOAR).D_95);
% display(qi_coarse(ixOAR).D_95);
% 
% 
% %% 
% % Export the dose in binary format. matRad can write multiple formats, here we 
% % choose to export in nrrd format using the matRad_writeCube function, which 
% % chooses the appropriate subroutine from the extension. The metadata struct can
% % also store more optional parameters, but requires only the resolution to be s
% % set.
% metadata = struct('resolution',[ct.resolution.x ct.resolution.y ct.resolution.z]);
% matRad_writeCube([pwd filesep 'photonDose_example2.nrrd'],resultGUI.physicalDose,'double',metadata);


