function sort_data(data_files)
    % Function to sort text files into shared position folders (p1-p?)
    % data_files: cell array of filenames (e.g., {'sSp1r1.txt', 'sFp2r3.txt', ...})
    
    % Create a shared output directory
    output_dir = fullfile(pwd, 'sorted_data');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Loop through each file and sort by position (pX)
    for i = 1:length(data_files)
        file_name = data_files{i};
        
        % Extract the position number from the filename (e.g., "p1" from sSp1rY or sFp1rY)
        [~, name, ~] = fileparts(file_name);
        % The regexp now matches files that start with "s", followed by either "S" or "F",
        % then "p" and the position number, then "r" followed by the run number.
        parts = regexp(name, 's[FLRT]p(\d+)r\d+', 'tokens');
        
        if ~isempty(parts)
            position_num = parts{1}{1}; % Extract the captured digits for the position
            position_folder = fullfile(output_dir, ['p', position_num]);
            
            % Create the position folder if it doesn't exist
            if ~exist(position_folder, 'dir')
                mkdir(position_folder);
            end
            
            % Move the file to the appropriate position folder
            movefile(file_name, fullfile(position_folder, file_name));
        end
    end
end
