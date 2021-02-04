COCO_COLORS = [(255, 0, 0),
               (255, 85, 0),
               (255, 170, 0),
               (255, 255, 0),
               (170, 255, 0),
               (85, 255, 0),
               (0, 255, 0),
               (0, 255, 85),
               (0, 255, 170),
               (0, 255, 255),
               (0, 170, 255),
               (0, 85, 255),
               (0, 0, 255),
               (50, 0, 255),
               (100, 0, 255),
               (170, 0, 255),
               (255, 0, 255),
               (255, 150, 0),
               (85, 170, 0),
               (42, 128, 85),
               (0, 85, 170),
               (255, 0, 170),
               (255, 0, 85),
               (242, 165, 65)]

# JOINT_IDS_FOR_ESTIMATION_DATA
# Nose = 0,
# Neck = 1,
# RShoulder = 2,
# RElbow = 3,
# RWrist = 4,
# LShoulder = 5,
# LElbow = 6,
# LWrist = 7,
# MidHip = 8,  Corresponds to Root in Human36M joints
# RHip = 9,
# RKnee = 10,
# RAnkle = 11,
# LHip = 12,
# LKnee = 13,
# LAnkle = 14,
# REye = 15,  Not used in estimation
# LEye = 16,  Not used in estimation
# REar = 17,  Not used in estimation
# LEar = 18,  Not used in estimation
# Head = 19,
# Belly = 20,

CONNECTED_JOINTS_PAIRS_FOR_ESTIMATION = [(8, 9),
                                         (9, 10),
                                         (10, 11),
                                         (8, 12),
                                         (12, 13),
                                         (13, 14),
                                         (8, 20),
                                         (1, 20),
                                         (0, 1),
                                         (0, 19),
                                         (1, 2),
                                         (2, 3),
                                         (3, 4),
                                         (1, 5),
                                         (5, 6),
                                         (6, 7)]

# JOINT_IDS_FOR_HUMAN36M_GROUND_TRUTH_DATA
# Root = 0  Corresponds to MidHip in estimated joints
# RHip = 1
# RKnee = 2
# RAnkle = 3
# LHip = 4
# LKnee = 5
# LAnkle = 6
# Belly = 7
# Neck = 8
# Nose = 9
# Head = 10
# LShoulder = 11
# LElbow = 12
# LWrist = 13
# RShoulder = 14
# RElbow = 15
# RWrist = 16

CONNECTED_JOINTS_PAIRS_FOR_HUMAN36M_GROUND_TRUTH = [(0, 1),
                                                    (1, 2),
                                                    (2, 3),
                                                    (0, 4),
                                                    (4, 5),
                                                    (5, 6),
                                                    (0, 7),
                                                    (7, 8),
                                                    (8, 9),
                                                    (9, 10),
                                                    (8, 11),
                                                    (11, 12),
                                                    (12, 13),
                                                    (8, 14),
                                                    (14, 15),
                                                    (15, 16)]

JOINT_MAPPING_FROM_GT_TO_ESTIMATION = [(0, 8),
                                       (1, 9),
                                       (2, 10),
                                       (3, 11),
                                       (4, 12),
                                       (5, 13),
                                       (6, 14),
                                       (7, 20),
                                       (8, 1),
                                       (9, 0),
                                       (10, 19),
                                       (11, 5),
                                       (12, 6),
                                       (13, 7),
                                       (14, 2),
                                       (15, 3),
                                       (16, 4)]
