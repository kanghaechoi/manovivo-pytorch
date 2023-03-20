%% Initialize
clear;
clc;

%% 손목각도 변환

% Select files
PROMPT_QUESTION = 'Insert question number (Options: q2, q3): ';
QUESTION = input(PROMPT_QUESTION, 's');

AGE = 20;

PROMPT_CLASS = 'Insert binary class (Options: 0, 1): ';
CLASS = input(PROMPT_CLASS);

hand_count = numel(dir(sprintf('../data/%s/Hand_IMU_%d_%d_*.txt', ... 
    QUESTION, AGE, CLASS)));
wrist_count = numel(dir(sprintf('../data/%s/Wrist_IMU_%d_%d_*.txt', ...
    QUESTION, AGE, CLASS)));

if hand_count ~= wrist_count
    file_count = min([hand_count, wrist_count]);
else
    file_count = hand_count;
end

for i=1:file_count
    % Read hand data
    if i < 10
        hand_data = readmatrix(sprintf('../data/%s/Hand_IMU_%d_%d_0%d.txt', ...
            QUESTION, AGE, CLASS, i));
        sprintf('../data/%s/Hand_IMU_%d_%d_0%d.txt', ...
            QUESTION, AGE, CLASS, i)
    else
        hand_data = readmatrix(sprintf('../data/%s/Hand_IMU_%d_%d_%d.txt', ...
            QUESTION, AGE, CLASS, i));
        sprintf('../data/%s/Hand_IMU_%d_%d_%d.txt', ...
            QUESTION, AGE, CLASS, i)
    end   

    % Read wrist
    if i < 10
        wrist_data = readmatrix(sprintf('../data/%s/Wrist_IMU_%d_%d_0%d.txt', ...
            QUESTION, AGE, CLASS, i));
        sprintf('../data/%s/Wrist_IMU_%d_%d_0%d.txt', ...
            QUESTION, AGE, CLASS, i)
    else
        wrist_data = readmatrix(sprintf('../data/%s/Wrist_IMU_%d_%d_%d.txt', ...
            QUESTION, AGE, CLASS, i));
        sprintf('../data/%s/Wrist_IMU_%d_%d_%d.txt', ...
            QUESTION, AGE, CLASS, i)
    end

    if length(hand_data) >= length(wrist_data)
        feature_length = length(wrist_data);
    elseif length(hand_data) < length(wrist_data)
        feature_length = length(hand_data);
    end

    Hand_phi1 = hand_data(1:feature_length,1);
    Hand_theta1 = hand_data(1:feature_length,2);
    Hand_psi1 = hand_data(1:feature_length,3);
    Wrist_phi1 = wrist_data(1:feature_length,1);
    Wrist_theta1 = wrist_data(1:feature_length,2);
    Wrist_psi1 = wrist_data(1:feature_length,3);



    % 각 데이터의 길이를 맞춘 상태로 시작
    phi_upper=deg2rad(Wrist_phi1);
    theta_upper=deg2rad(Wrist_theta1);
    psi_upper=deg2rad(Wrist_psi1);

    phi_lower=deg2rad(Hand_phi1);
    theta_lower=deg2rad(Hand_theta1);
    psi_lower=deg2rad(Hand_psi1);

    % upper body segment(손목센서) euler angle의 rotation matrix
    R_F(1,1,:) = cos(psi_upper).*cos(theta_upper);
    R_F(1,2,:) = cos(psi_upper).*sin(theta_upper).*sin(phi_upper) + sin(psi_upper).*cos(phi_upper);
    R_F(1,3,:) = - cos(psi_upper).*sin(theta_upper).*cos(phi_upper) + sin(psi_upper).*sin(phi_upper) ;

    R_F(2,1,:) = -sin(psi_upper).*cos(theta_upper);
    R_F(2,2,:) = - sin(psi_upper).*sin(theta_upper).*sin(phi_upper) + cos(psi_upper).*cos(phi_upper) ;
    R_F(2,3,:) = sin(psi_upper).*sin(theta_upper).*cos(phi_upper) + cos(psi_upper).*sin(phi_upper);

    R_F(3,1,:) = sin(theta_upper);
    R_F(3,2,:) = -cos(theta_upper).*sin(phi_upper);
    R_F(3,3,:) = cos(theta_upper).*cos(phi_upper);

    % lower body segment(손등센서) euler angle의 rotation matrix
    R_T(1,1,:) = cos(psi_lower).*cos(theta_lower);
    R_T(1,2,:) = cos(psi_lower).*sin(theta_lower).*sin(phi_lower) + sin(psi_lower).*cos(phi_lower);
    R_T(1,3,:) = -cos(psi_lower).*sin(theta_lower).*cos(phi_lower) + sin(psi_lower).*sin(phi_lower) ;

    R_T(2,1,:) = -sin(psi_lower).*cos(theta_lower);
    R_T(2,2,:) = -sin(psi_lower).*sin(theta_lower).*sin(phi_lower) + cos(psi_lower).*cos(phi_lower);
    R_T(2,3,:) = sin(psi_lower).*sin(theta_lower).*cos(phi_lower) + cos(psi_lower).*sin(phi_lower);

    R_T(3,1,:) = sin(theta_lower);
    R_T(3,2,:) = -cos(theta_lower).*sin(phi_lower);
    R_T(3,3,:) = cos(theta_lower).*cos(phi_lower);

    % 2개의 rotation matrix를 하나의 rotation matrix로 합침
    for j=1:length(R_T)
        R_two_segment(:,:,j)=R_T(:,:,j)*R_F(:,:,j)';
    end

    % rotation matrix -> joint angle로 변환
    alpha=atan2(-R_two_segment(3,2,:),R_two_segment(3,3,:));
    beta=atan2(R_two_segment(3,1,:),sqrt(R_two_segment(1,1,:).^2 + R_two_segment(1,2,:).^2));
    gamma=atan2(-R_two_segment(2,1,:),R_two_segment(1,1,:));

    %Flexion_extention=radtodeg(alpha);
    joint_phi=rad2deg(alpha);
    joint_phi=joint_phi(1,:);

    %abduction_adduction=radtodeg(beta);
    joint_theta=rad2deg(beta);
    joint_theta=joint_theta(1,:);

    %Internal_external_rotation=radtodeg(gamma);
    joint_psi=rad2deg(gamma);
    joint_psi=joint_psi(1,:);

    helical_data = round([joint_phi; joint_theta; joint_psi]', 6);

    if i < 10
        writematrix(helical_data, sprintf('../data/%s/Helical_IMU_%d_%d_0%d.txt', ...
            QUESTION, AGE, CLASS, i), 'Delimiter', 'space');
        sprintf('../data/%s/Helical_IMU_%d_%d_0%d.txt', ...
            QUESTION, AGE, CLASS, i)
    else
        writematrix(helical_data, sprintf('../data/%s/Helical_IMU_%d_%d_%d.txt', ...
            QUESTION, AGE, CLASS, i), 'Delimiter', 'space');
        sprintf('../data/%s/Helical_IMU_%d_%d_%d.txt', ...
            QUESTION, AGE, CLASS, i)
    end

    clearvars i j
    clearvars R_F R_T
    clearvars alpha beta gamma
    clearvars joint_phi joint_theta joint_psi
end