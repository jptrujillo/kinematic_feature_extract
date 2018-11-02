%%Plots and saves stick-light ("skeleton") on a black background, centered in the frame
y = i;


figure('Color',[0 0 0]);
% if exist('LT','var'),
%     vidObj = VideoWriter(['./skeletons_whole/fMRI_pilot/' num2str(x) '_' num2str(y) '_' num2str(LT+1) '_skeleton.avi']); open(vidObj);
% else
%     vidObj = VideoWriter(['./skeletons_whole/fMRI_pilot/' num2str(x) '_' num2str(y) '_skeleton.avi']); open(vidObj);
% end
vidObj = VideoWriter(['C:/Users/James/Documents/Matlab/skeleton_rotate_L.avi']);
open(vidObj);
%dynamic struct field reference
jx = cell(1,25);
for i = 1:25,
    jx(i) = {strcat('j',  num2str(i-1))};
end

%center skeleton based on mid-spine
C = Trial.j1(1,2:4);
for ii = 1:25, 
    for i = 1:length(Trial.(jx{ii})),
    Trial.(jx{ii})(i,2:4) = (Trial.(jx{ii})(i,2:4)-C);
    end
end
% 
% Trial2 = Trial;
% %find angle diff between upper and lower spine
% %FIRST ROTATE SPINE TO ALIGN Y-AXIS
% %to align TO the y-axis, you must rotate ALONG the z-axis (so z-stays the
% %same
% %Trial.j1 %base
% %Trial.j20 %shoulder
% t = 0:0.1:1;
% C = repmat(Trial.j1(1,2:4),length(t),1)'+(Trial.j20(1,2:4)-Trial.j1(1,2:4))'*t; %current orientation
% DE = sqrt(((Trial.j1(1,2)-Trial.j20(1,2)).^2)+((Trial.j1(1,3)-Trial.j20(1,3)).^2)+((Trial.j1(1,4)-Trial.j20(1,4)).^2)); %length from shoulder to base of spine
% C2 = repmat(Trial.j20(1,2:4),length(t),1)'+([Trial.j20(1,2) Trial.j20(1,3)-DE (Trial.j20(1,4))]-Trial.j20(1,2:4))'*t;
% vertrot = atan2(norm(cross(C,C2)), dot(C,C2));
% center = repmat([C(1,5); C(2,5)],1,length(C));
% % theta = pi/3;
% theta = vertrot(length(vertrot));
% R = [cos(theta) -sin(theta); sin(theta) cos(theta)]; %rotation matrix (around 1 axis)
% clear v vo 
% for i = 1:length(Trial.j1(:,1)),%for each time point
%     for ii = 1:25,
%         v(1:2) = [Trial.(jx{ii})(i,2); Trial.(jx{ii})(i,3)];
%         vo(1:2) = R*(v' - center(:,1)) + center(:,1);
%         Trial2.(jx{ii})(i,2) = vo(1);
%         Trial2.(jx{ii})(i,3) = vo(2);
%     end
%     
% end
% Trial = Trial2;
% 
% 
% %NEXT ALIGN SHOULDERS ONTO X-AXIS
% C =repmat(Trial.j4(1,2:4),length(t),1)'+(Trial.j8(1,2:4)-Trial.j4(1,2:4))'*t;
% DE = sqrt(((Trial.j4(1,2)-Trial.j8(1,2)).^2)+((Trial.j4(1,3)-Trial.j8(1,3)).^2)+((Trial.j4(1,4)-Trial.j8(1,4)).^2)); %length from shoulder to shoulder
% C2 = repmat(Trial.j8(1,2:4),length(t),1)'+([Trial.j8(1,2)-DE Trial.j8(1,3) (Trial.j8(1,4))]-Trial.j8(1,2:4))'*t;
% vertrot = atan2(norm(cross(C,C2)), dot(C,C2));
% center = repmat([C(1,5); C(2,5)],1,length(C));
% % theta = pi/3;
% theta = vertrot(length(vertrot));
% R = [cos(theta) -sin(theta); sin(theta) cos(theta)]; %rotation matrix (around 1 axis)
% clear v vo 
% 
% for i = 1:length(Trial.j1(:,1)),%for each time point
%     for ii = 1:25,
%         v(1:2) = [Trial.(jx{ii})(i,2); Trial.(jx{ii})(i,4)];
%         vo(1:2) = R*(v' - center(:,1)) + center(:,1);
%         Trial2.(jx{ii})(i,2) = vo(1);
%         Trial2.(jx{ii})(i,4) = vo(2);
%     end
%     
% end
% Trial = Trial2;
% 
% %ALIGN SPINE TO BE AT X=X AT BOTH POINTS
% t = 0:0.1:1;
% C = repmat(Trial.j1(1,2:4),length(t),1)'+(Trial.j20(1,2:4)-Trial.j1(1,2:4))'*t; %current orientation
% DE = sqrt(((Trial.j1(1,2)-Trial.j20(1,2)).^2)+((Trial.j1(1,3)-Trial.j20(1,3)).^2)+((Trial.j1(1,4)-Trial.j20(1,4)).^2)); %length from shoulder to base of spine
% 
% center = repmat([C(1,5); C(2,5)],1,length(C)); %in C2, both x-axis coordinates must match this x-coordinate
% 
% C2 = repmat([center(1,1) Trial.j20(1,3) Trial.j20(1,4)],length(t),1)'+([center(1,1) Trial.j20(1,3)-DE (Trial.j20(1,4))]-[center(1,1) Trial.j20(1,3) Trial.j20(1,4)])'*t;
% % vertrot = atan2(norm(cross(C2,C), dot(C2,C)));
% vertrot2 = atan2((C2(2,1)-C(2,1)),(C2(1,1)-C(1,1)));
% % theta = pi/3;
% theta = max(vertrot2);
% R = [cos(theta) -sin(theta); sin(theta) cos(theta)]; %rotation matrix (around 1 axis)
% clear v vo 
% for i = 1:length(Trial.j1(:,1)),%for each time point
%     for ii = 1:25,
%         v(1:2) = [Trial.(jx{ii})(i,2); Trial.(jx{ii})(i,3)];
%         vo(1:2) = R*(v' - center(:,1)) + center(:,1);
%         Trial2.(jx{ii})(i,2) = vo(1);
%         Trial2.(jx{ii})(i,3) = vo(2);
%     end
%     
% end
% Trial = Trial2;
% 
% %NOW ALIGN SPINE TO Z-AXIS
% C = repmat(Trial.j1(1,2:4),length(t),1)'+(Trial.j20(1,2:4)-Trial.j1(1,2:4))'*t; %current orientation
% DE = sqrt(((Trial.j1(1,2)-Trial.j20(1,2)).^2)+((Trial.j1(1,3)-Trial.j20(1,3)).^2)+((Trial.j1(1,4)-Trial.j20(1,4)).^2)); %length from shoulder to base of spine
% C2 = repmat(Trial.j20(1,2:4),length(t),1)'+([Trial.j20(1,2) Trial.j20(1,3) (Trial.j20(1,4)-Trial.j20(1,4))]-Trial.j20(1,2:4))'*t;
% vertrot = atan2(norm(cross(C,C2)), dot(C,C2));
% center = repmat([C(1,5); C(2,5)],1,length(C));
% % theta = pi/3;
% theta = max(vertrot);
% R = [cos(theta) -sin(theta); sin(theta) cos(theta)]; %rotation matrix (around 1 axis)
% clear v vo 
% Trial2 = Trial;
% for i = 1:length(Trial.j1(:,1)),%for each time point
%     for ii = 1:25,
%         v(1:2) = [Trial.(jx{ii})(i,3); Trial.(jx{ii})(i,4)];
%         vo(1:2) = R*(v' - center(:,1)) + center(:,1);
%         Trial2.(jx{ii})(i,3) = vo(1);
%         Trial2.(jx{ii})(i,4) = vo(2);
%     end
%     
% end
% 
% Trial = Trial2;
% 
% %TRY SINGLE ROTATIONS?
% 
% %Potshot
% % %REALIGN SPINE TO Z-AXIS
% % C = repmat(Trial.j1(1,2:4),length(t),1)'+(Trial.j20(1,2:4)-Trial.j1(1,2:4))'*t; %current orientation
% % DE = sqrt(((Trial.j1(1,2)-Trial.j20(1,2)).^2)+((Trial.j1(1,3)-Trial.j20(1,3)).^2)+((Trial.j1(1,4)-Trial.j20(1,4)).^2)); %length from shoulder to base of spine
% % C2 = repmat(Trial.j20(1,2:4),length(t),1)'+([Trial.j20(1,2) Trial.j20(1,3) (Trial.j20(1,4)-Trial.j20(1,4))]-Trial.j20(1,2:4))'*t;
% % vertrot = atan2(norm(cross(C,C2)), dot(C,C2));
% % center = repmat([C(1,5); C(2,5)],1,length(C));
% % % theta = pi/3;
% % theta = vertrot(length(vertrot));
% % R = [cos(theta) -sin(theta); sin(theta) cos(theta)]; %rotation matrix (around 1 axis)
% % clear v vo 
% % Trial2 = Trial;
% % for i = 1:length(Trial.j1(:,1)),%for each time point
% %     for ii = 1:25,
% %         v(1:2) = [Trial.(jx{ii})(i,3); Trial.(jx{ii})(i,4)];
% %         vo(1:2) = R*(v' - center(:,1)) + center(:,1);
% %         Trial2.(jx{ii})(i,3) = vo(1);
% %         Trial2.(jx{ii})(i,4) = vo(2);
% %     end
% %     
% % end
% 
% Trial = Trial2;
%DO THIS SEPARATELY FOR EVERY ROTATION ANGLE UNTIL WE FIND ONE THAT LETS
%AZIMUTH ROTATE AROUND X AXIS


% v01 = v(:,10);
% vo1 = R*(v01 - center(:,1)) + center(:,1);
% % v = [C(1,:); C(3,:)];
% % s = v - center;
% vo = R*(v - center) + center; 

for IX=1:size(Trial.j7,1)
    %left arm
    plot3([Trial.j6(IX,2) Trial.j7(IX,2)],[Trial.j6(IX,3) Trial.j7(IX,3)],[Trial.j6(IX,4) Trial.j7(IX,4)],'g');
    hold on;
%L side
    az = 90; el=0;
    view(az,el)

    
    
    %left arm
    plot3([Trial.j4(IX,2) Trial.j20(IX,2)],[Trial.j4(IX,3) Trial.j20(IX,3)],[Trial.j4(IX,4) Trial.j20(IX,4)],'LineWidth',1.5,'color','g')%Spine shoulder - L shoulder
    plot3([Trial.j4(IX,2) Trial.j5(IX,2)],[Trial.j4(IX,3) Trial.j5(IX,3)],[Trial.j4(IX,4) Trial.j5(IX,4)],'LineWidth',1.5,'color','g');%L shoulder - L elbow
    plot3([Trial.j6(IX,2) Trial.j5(IX,2)],[Trial.j6(IX,3) Trial.j5(IX,3)],[Trial.j6(IX,4) Trial.j5(IX,4)],'LineWidth',1.5,'color','g'); %L elbow - L wrist
    plot3([Trial.j6(IX,2) Trial.j7(IX,2)],[Trial.j6(IX,3) Trial.j7(IX,3)],[Trial.j6(IX,4) Trial.j7(IX,4)],'LineWidth',1.5,'color','g'); %L wrist - L hand
  
    %right arm
    plot3([Trial.j8(IX,2) Trial.j20(IX,2)],[Trial.j8(IX,3) Trial.j20(IX,3)],[Trial.j8(IX,4) Trial.j20(IX,4)],'LineWidth',1.5,'color','g');%Spine shoulder - R shoulder
    plot3([Trial.j8(IX,2) Trial.j9(IX,2)],[Trial.j8(IX,3) Trial.j9(IX,3)],[Trial.j8(IX,4) Trial.j9(IX,4)],'LineWidth',1.5,'color','g'); %R shoulder - R elbow
    plot3([Trial.j10(IX,2) Trial.j9(IX,2)],[Trial.j10(IX,3) Trial.j9(IX,3)],[Trial.j10(IX,4) Trial.j9(IX,4)],'LineWidth',1.5,'color','g'); %R elbow - R wrist
    plot3([Trial.j10(IX,2) Trial.j11(IX,2)],[Trial.j10(IX,3) Trial.j11(IX,3)],[Trial.j10(IX,4) Trial.j11(IX,4)],'LineWidth',1.5,'color','g');%R wrist - R hand

    
    %remain
    plot3([Trial.j1(IX,2) Trial.j0(IX,2)],[Trial.j1(IX,3) Trial.j0(IX,3)],[Trial.j1(IX,4) Trial.j0(IX,4)],'LineWidth',1.5,'color','g')%Spine base - spine mid
    plot3([Trial.j1(IX,2) Trial.j20(IX,2)],[Trial.j1(IX,3) Trial.j20(IX,3)],[Trial.j1(IX,4) Trial.j20(IX,4)],'LineWidth',1.5,'color','g')%Spine mid - spine shoulder
    plot3([Trial.j2(IX,2) Trial.j20(IX,2)],[Trial.j2(IX,3) Trial.j20(IX,3)],[Trial.j2(IX,4) Trial.j20(IX,4)],'LineWidth',1.5,'color','g')%Spine shoulder - neck
    plot3([Trial.j2(IX,2) Trial.j3(IX,2)],[Trial.j2(IX,3) Trial.j3(IX,3)],[Trial.j2(IX,4) Trial.j3(IX,4)],'LineWidth',1.5,'color','g')%Neck - Head
   
    

    axis([-0.5 0.5 -0.5 0.5 -1 1])
    % xlabel('X-axis');ylabel('Y-axis');zlabel('Z-axis');
    ax=gca;
    set(ax,'Visible','off');
    plot3([Trial.j6(IX,2) Trial.j7(IX,2)],[Trial.j6(IX,3) Trial.j7(IX,3)],[Trial.j6(IX,4) Trial.j7(IX,4)],'g');
    hold off;


    
pause(0.0333)
F = getframe(gcf);
writeVideo(vidObj,F);
end
hold off
figure('Color',[0 0 0]);
vidObj = VideoWriter(['C:/Users/James/Documents/Matlab/skeleton_rotate_R.avi']);
open(vidObj);
for IX=1:size(Trial.j7,1),
%R side
    az = -90; el=0;
    view(az,el)

    
    
    %left arm
    plot3([Trial.j4(IX,2) Trial.j20(IX,2)],[Trial.j4(IX,3) Trial.j20(IX,3)],[Trial.j4(IX,4) Trial.j20(IX,4)],'LineWidth',1.5,'color','g')%Spine shoulder - L shoulder
    hold on
    plot3([Trial.j4(IX,2) Trial.j5(IX,2)],[Trial.j4(IX,3) Trial.j5(IX,3)],[Trial.j4(IX,4) Trial.j5(IX,4)],'LineWidth',1.5,'color','g');%L shoulder - L elbow
    plot3([Trial.j6(IX,2) Trial.j5(IX,2)],[Trial.j6(IX,3) Trial.j5(IX,3)],[Trial.j6(IX,4) Trial.j5(IX,4)],'LineWidth',1.5,'color','g'); %L elbow - L wrist
    plot3([Trial.j6(IX,2) Trial.j7(IX,2)],[Trial.j6(IX,3) Trial.j7(IX,3)],[Trial.j6(IX,4) Trial.j7(IX,4)],'LineWidth',1.5,'color','g'); %L wrist - L hand
  
    %right arm
    plot3([Trial.j8(IX,2) Trial.j20(IX,2)],[Trial.j8(IX,3) Trial.j20(IX,3)],[Trial.j8(IX,4) Trial.j20(IX,4)],'LineWidth',1.5,'color','g');%Spine shoulder - R shoulder
    plot3([Trial.j8(IX,2) Trial.j9(IX,2)],[Trial.j8(IX,3) Trial.j9(IX,3)],[Trial.j8(IX,4) Trial.j9(IX,4)],'LineWidth',1.5,'color','g'); %R shoulder - R elbow
    plot3([Trial.j10(IX,2) Trial.j9(IX,2)],[Trial.j10(IX,3) Trial.j9(IX,3)],[Trial.j10(IX,4) Trial.j9(IX,4)],'LineWidth',1.5,'color','g'); %R elbow - R wrist
    plot3([Trial.j10(IX,2) Trial.j11(IX,2)],[Trial.j10(IX,3) Trial.j11(IX,3)],[Trial.j10(IX,4) Trial.j11(IX,4)],'LineWidth',1.5,'color','g');%R wrist - R hand

    
    %remain
    plot3([Trial.j1(IX,2) Trial.j0(IX,2)],[Trial.j1(IX,3) Trial.j0(IX,3)],[Trial.j1(IX,4) Trial.j0(IX,4)],'LineWidth',1.5,'color','g')%Spine base - spine mid
    plot3([Trial.j1(IX,2) Trial.j20(IX,2)],[Trial.j1(IX,3) Trial.j20(IX,3)],[Trial.j1(IX,4) Trial.j20(IX,4)],'LineWidth',1.5,'color','g')%Spine mid - spine shoulder
    plot3([Trial.j2(IX,2) Trial.j20(IX,2)],[Trial.j2(IX,3) Trial.j20(IX,3)],[Trial.j2(IX,4) Trial.j20(IX,4)],'LineWidth',1.5,'color','g')%Spine shoulder - neck
    plot3([Trial.j2(IX,2) Trial.j3(IX,2)],[Trial.j2(IX,3) Trial.j3(IX,3)],[Trial.j2(IX,4) Trial.j3(IX,4)],'LineWidth',1.5,'color','g')%Neck - Head
   
    axis([-0.5 0.5 -0.5 0.5 -1 1])
    % xlabel('X-axis');ylabel('Y-axis');zlabel('Z-axis');
    ax=gca;
    set(ax,'Visible','off');
    hold off
    pause(0.0333)
F = getframe(gcf);
writeVideo(vidObj,F);
end
close(vidObj);