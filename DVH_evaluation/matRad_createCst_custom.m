function cst = matRad_createCst_custom(structures, patient_file_name)
% matRad function to create a cst struct upon dicom import
% 
% call
%   cst = matRad_createCst(structures)
%
% input
%   structures:     matlab struct containing information about rt structure
%                   set (generated with matRad_importDicomRtss and 
%                   matRad_convRtssContours2Indices)
%   patient_file_name:        patient name for the specific target rules (incl .mat)
%
% output
%   cst:            matRad cst struct
%
% References
%   -
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright 2015 the matRad development team. 
% 
% This file is part of the matRad project. It is subject to the license 
% terms in the LICENSE file found in the top-level directory of this 
% distribution and at https://github.com/e0404/matRad/LICENSES.txt. No part 
% of the matRad project, including this file, may be copied, modified, 
% propagated, or distributed except according to the terms contained in the 
% LICENSE file.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

matRad_cfg = MatRad_Config.instance();
patient=convertCharsToStrings(patient_file_name(1:(length(patient_file_name)-4)));
nStructures = size(structures,2);
cst = cell(nStructures,6);

%Create set of default colors
defaultColors = colorcube(nStructures);

for i = 1:size(structures,2)
    cst{i,1} = i - 1; % first organ has number 0    
    cst{i,2} = structures(i).structName;
    cst{i,4}{1} = structures(i).indices;


    if isempty(cst{i,4}{1})
            cst{i,3} = 'IGNORED';
            cst{i,5}.Priority = 3;
            cst{i,6} = []; % define no OAR dummy objcetives  
        
    else


        if ~isempty(regexpi(cst{i,2},'gtv_ph'))&& ...
           isempty(regexpi(cst{i,2},'.+gtv_ph'))&& ...
           isempty(regexpi(cst{i,2},'.+gtv_ph.+'))&& ...
           isempty(regexpi(cst{i,2},'gtv_ph.+')) 
            
            cst{i,3} = 'TARGET'; 
            cst{i,5}.Priority = 1;     
            % default objectives for targets
            objective = DoseObjectives.matRad_MinDVH;
            objective.penalty = 1;
            objective.parameters = {54,95};  %dose, to volume
            cst{i,6}{1} = struct(objective);
    
     
        elseif ((~isempty(regexpi(cst{i,2},'gtv1_v1_1a'))&& ~isempty(regexpi(patient,"PATIENT_LIST")))
            
            cst{i,3} = 'TARGET'; 
            cst{i,5}.Priority = 1;     
            % default objectives for targets
            objective = DoseObjectives.matRad_MinDVH;
            objective.penalty = 1;
            objective.parameters = {54,95};  %dose, to volume
            cst{i,6}{1} = struct(objective);
        
        elseif ~isempty(regexpi(cst{i,2},'ptv_ph'))&& ...
           isempty(regexpi(cst{i,2},'ptv_ph.+')) 
            
            cst{i,3} = 'TARGET'; 
            cst{i,5}.Priority = 1;     
            % default objectives for targets
            objective = DoseConstraints.matRad_MinMaxDose;
            objective.parameters = {40,62.4,1};  %min,max,method'approx'
            cst{i,6}{1} = struct(objective);
    
         elseif ((~isempty(regexpi(cst{i,2},'ptv1_v1_1a'))&& ~isempty(regexpi(patient,"PATIENT_LIST")))
    
            cst{i,3} = 'TARGET'; 
            cst{i,5}.Priority = 1;     
            % default objectives for targets
            objective = DoseConstraints.matRad_MinMaxDose;
            objective.parameters = {40,62.4,1};  %min,max,method'approx'
            cst{i,6}{1} = struct(objective);
    

            elseif (~isempty(regexpi(cst{i,2},'stomach'))&& isempty(regexpi(patient,"PATIENT_LIST"))) || ...
           (~isempty(regexpi(cst{i,2},'duodenum')) && isempty(regexpi(patient,"PATIENT_LIST"))) || ...
           (~isempty(regexpi(cst{i,2},'bowel'))&&isempty(regexpi(cst{i,2},'bowel.+'))&& isempty(regexpi(patient,"PATIENT_LIST")))
            
            cst{i,3} = 'OAR';
            cst{i,5}.Priority = 2;
                    % default objectives for targets
            objective = DoseObjectives.matRad_MaxDVH;
            objective.penalty = 1;
            objective.parameters = {30,1};  %dose, to volume
            cst{i,6}{1} = struct(objective);
    
        elseif ~isempty(regexpi(cst{i,2},'spinal'))
            
            cst{i,3} = 'OAR';
            cst{i,5}.Priority = 2;
                    % default objectives for targets
            objective = DoseObjectives.matRad_MaxDVH;
            objective.penalty = 1;
            objective.parameters = {12,1};  %dose, to volume
            cst{i,6}{1} = struct(objective);
    
         elseif (~isempty(regexpi(cst{i,2},'liver')) && ...
           isempty(regexpi(cst{i,2},'liver_ph'))&& ...
           isempty(regexpi(cst{i,2},'.+liver'))) 
            
            cst{i,3} = 'OAR';
            cst{i,5}.Priority = 2;
                    % default objectives for targets
            objective = DoseObjectives.matRad_MeanDose;
            objective.penalty = 1;
            objective.parameters = {18};  %dose
            cst{i,6}{1} = struct(objective);
    
        else
            cst{i,3} = 'IGNORED';
            cst{i,5}.Priority = 3;
            cst{i,6} = []; % define no OAR dummy objcetives  
        
        end
    end


    
    % set default parameter for biological planning
    cst{i,5}.alphaX  = 0.1;
    cst{i,5}.betaX   = 0.05;
    cst{i,5}.Visible = 1;
    if isfield(structures(i),'structColor') && ~isempty(structures(i).structColor)
        cst{i,5}.visibleColor = structures(i).structColor' ./ 255;
    else
        cst{i,5}.visibleColor = defaultColors(i,:);
        matRad_cfg.dispInfo('No color information for structure %d "%s". Assigned default color [%f %f %f]\n',i,cst{i,2},defaultColors(i,1),defaultColors(i,2),defaultColors(i,3));
    end
end
