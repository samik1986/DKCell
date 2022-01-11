imshow(x221, [0 4000])
hold on;
fid = fopen('DK39_CH3_premotor.csv');

while ~feof(fid)
    tline = fgetl(fid);
    locs = strfind(tline, ',');
    filename = tline(1:locs(1)-1);
    xPos = str2num(tline(locs(1)+1:locs(2)-1));
    yPos = str2num(tline(locs(2)+1:end));
    plot(xPos, yPos, 'g.');
end
fclose(fid);