clear
% NN_results_folder="<PATH>\2nd_paper_pix2pix_no_norm_2d_finetuned";


NN_results_folder_common="<PATH>/dose_calcs/pix2pix";
STRUCT_folder_initial="<PATH>/initial"

matRadGUI
model_results_folders=["MODEL LIST"];


for model_results_folder = model_results_folders
    NN_results_folder=fullfile(NN_results_folder_common,model_results_folder);
    subdirs=["fake","real_overwritten"];
    for subdir = subdirs
        D = dir(fullfile(NN_results_folder,subdir)); % A is a struct ... first elements are '.' and '..' used for navigation.
        for k = 3:length(D) % avoid using the first ones
            currD = D(k).name;
            files=[];
            patient=currD;
            dir_path_field = fullfile(NN_results_folder,subdir, patient);
            cd(fullfile(NN_results_folder,subdir,currD)) 
            [fileList, patient_listbox] = matRad_scanDicomImportFolder(dir_path_field);
            
            ctseries_listbox=1;
            ctseries_listbox=string(unique(fileList(strcmp(fileList(:,2), 'CT'),4)));
                    
            if ~isempty(ctseries_listbox)
                res_x = str2double(unique(fileList(strcmp(fileList(:,2), 'CT'),9)));
                res_y = str2double(unique(fileList(strcmp(fileList(:,2), 'CT'),10)));
                res_z = str2double(unique(fileList(strcmp(fileList(:,2), 'CT'),11)));
            else
                res_x = NaN; res_y = NaN; res_z = NaN;
            end
            
            % selected RT Plan
            %rtplan = fileList( strcmp(fileList(:,2),'RTPLAN'),:);
            %rtplan = string(fileList( strcmp(fileList(:,2),'RTPLAN'),1));
            %files.rtplan =rtplan;
            %files.rtplan =[];
            %rtdose = string(fileList( strcmp(fileList(:,2),'RTDOSE'),1));
            %files.rtdose =rtdose;
            %files.rtdose =[];
            files.resx =res_x;
            files.resy =res_y;
            files.resz = res_z;
            
    
            %!!!!!!!!!!!get struct file from the common folder!!!
    
            %allRtss = string(fileList( strcmp(fileList(:,2),'RTSTRUCT'),1));
    
            dir_path_field = fullfile(STRUCT_folder_initial, patient, "Plan");
            %cd(fullfile(NN_results_folder,subdir,currD)) 
            struct = sprintf('RTSTRUCT*.dcm');
            S = dir(fullfile(dir_path_field,struct));
            allRtss =string(fullfile( S.folder,S.name));
    
            files.rtss =allRtss;
            
            files.ct = string(fileList( strcmp(fileList(:,2),'CT'),1));
            
            % check if we should use the dose grid resolution
            files.useDoseGrid = 0;
            
            
            
            output_dir_path_field = convertStringsToChars(fullfile(NN_results_folder,subdir+'_matrad'));
            output_file_name=convertStringsToChars(patient+".mat");
            
            matRad_importDicom_custom(files, 0,1,output_dir_path_field,output_file_name);
            
            %[ct, cst, pln, resultGUI] =matRad_importDicom_custom(files, 0,1,output_dir_path_field,output_file_name);
    
        end
    end
end
    
    

