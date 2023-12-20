import numpy as np
import math
import cv2
import mediapipe
import networkx
from scipy.spatial import distance

import cv2
import mediapipe
import networkx
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from scipy.spatial import distance
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh


class EmotionalMesh:
    def __init__(self, frame_shape):
        # Source frame
        self.frame_shape = frame_shape



        # Indexes of 'Emotional Mesh' in  Mediapipe
        self.indexes = [61, 291, 0, 17, 50, 280, 48, 4, 278,
                        206, 426, 133, 130, 159, 145, 362, 359, 386, 374, 122,
                        351, 46, 105, 107, 276, 334, 336]

        self.edges=[(130,46),(46,105),(105,107),(107,4),(107,122),(336,4),(336,351),(336,334),(334,276),(276,359),(359,386),(386,362),(362,374),(374,351),(130,159),(159,133),(133,145),(145,50),(145,122),(122,4),(351,4),(4,48),(4,278),(48,50),(48,206),(278,280),(280,374),(280,291),(278,426),(426,291),(4,291),(4,61),(50,61),(206,61),(61,0),(0,291),(291,17),(61,17)]

        # Coordinates of 'Emotional Mesh' - Tuple (x, y)
        self.coordinates = []

        # Relative coordinates in source frame of 'Emotional Mesh' - Tuple (x, y)
        self.rcoordinates = []

        # Bounding box in source frame
        # Format: [cx_min, cy_min, cx_max, cy_max]
        self.bounding_box = np.zeros(4)

        # Angles
        # Format: [angle1, angle2, angle3, ...]
        self.num_angles = 21
        self.angles = np.zeros((1, self.num_angles))

    def update_landmarks(self):
        height, width, _ = self.frame_shape.shape

        faceModule = mediapipe.solutions.face_mesh
        processedImage = faceModule.FaceMesh(static_image_mode=True).process(self.frame_shape)
        if (processedImage.multi_face_landmarks is None):
            return
        for index in self.indexes:
            landmark = processedImage.multi_face_landmarks[0].landmark[index]
            pos = (int(landmark.x * width), int(landmark.y * height))
            x=landmark.x
            y=landmark.y
            self.coordinates.append((x, y))
            self.rcoordinates.append(pos)
        #self.__calculate_angles(self)
       # self.__calculate_bounding_box(processedImage.multi_face_landmarks)

    def draw(self, frame):
        for coord in self.rcoordinates:
            cv2.circle(frame, (coord[0], coord[1]), 2, (0, 255, 0), -1)

    '''
    # Private methods ----------------------------------------------------------------
    def __distance(self, point1, point2):
        x0 = point1[0]
        y0 = point1[1]
        x1 = point2[0]
        y1 = point2[1]
        return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

    def __angle(self, point1, point2, point3):
        side1 = self.__distance(point2, point3)
        side2 = self.__distance(point1, point3)
        side3 = self.__distance(point1, point2)

        angle = math.degrees(math.acos((side1 ** 2 + side3 ** 2 - side2 ** 2) / (2 * side1 * side3)))
        return angle

    def __calculate_bounding_box(self, face_landmarks):
        h, w, _ = self.frame_shape
        cx_min = w
        cy_min = h
        cx_max = cy_max = 0
        for id, lm in enumerate(face_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            if cx < cx_min:
                cx_min = cx
            if cy < cy_min:
                cy_min = cy
            if cx > cx_max:
                cx_max = cx
            if cy > cy_max:
                cy_max = cy
        self.bounding_box = [cx_min, cy_min, cx_max, cy_max]

    def __calculate_angles(self):
        index = 0

        # Angle 0
        self.angles[0][index] = self.__angle(self.coordinates[7], self.coordinates[1],
                                             self.coordinates[2])
        index += 1
        # Angle 1
        self.angles[0][index] = self.__angle(self.coordinates[2], self.coordinates[1],
                                             self.coordinates[3])
        index += 1
        # Angle 2
        self.angles[0][index] = self.__angle(self.coordinates[0], self.coordinates[2],
                                             self.coordinates[1])
        index += 1
        # Angle 3
        self.angles[0][index] = self.__angle(self.coordinates[1], self.coordinates[7],
                                             self.coordinates[8])
        index += 1
        # Angle 4
        self.angles[0][index] = self.__angle(self.coordinates[0], self.coordinates[7],
                                             self.coordinates[1])
        index += 1
        # Angle 5
        self.angles[0][index] = self.__angle(self.coordinates[8], self.coordinates[5],
                                             self.coordinates[1])
        index += 1
        # Angle 6
        self.angles[0][index] = self.__angle(self.coordinates[8], self.coordinates[10],
                                             self.coordinates[1])
        index += 1
        # Angle 7
        self.angles[0][index] = self.__angle(self.coordinates[18], self.coordinates[5],
                                             self.coordinates[8])
        index += 1
        # Angle 8
        self.angles[0][index] = self.__angle(self.coordinates[8], self.coordinates[7],
                                             self.coordinates[20])
        index += 1
        # Angle 9
        self.angles[0][index] = self.__angle(self.coordinates[26], self.coordinates[7],
                                             self.coordinates[23])
        index += 1
        # Angle 10
        self.angles[0][index] = self.__angle(self.coordinates[7], self.coordinates[20],
                                             self.coordinates[18])
        index += 1
        # Angle 11
        self.angles[0][index] = self.__angle(self.coordinates[20], self.coordinates[18],
                                             self.coordinates[5])
        index += 1
        # Angle 12
        self.angles[0][index] = self.__angle(self.coordinates[17], self.coordinates[16],
                                             self.coordinates[18])
        index += 1
        # Angle 13
        self.angles[0][index] = self.__angle(self.coordinates[26], self.coordinates[25],
                                             self.coordinates[24])
        index += 1
        # Angle 14
        self.angles[0][index] = self.__angle(self.coordinates[20], self.coordinates[26],
                                             self.coordinates[25])
        index += 1
        # Angle 15
        self.angles[0][index] = self.__angle(self.coordinates[25], self.coordinates[24],
                                             self.coordinates[16])
        index += 1
        # Angle 16
        self.angles[0][index] = self.__angle(self.coordinates[24], self.coordinates[16],
                                             self.coordinates[17])
        index += 1
        # Angle 17
        self.angles[0][index] = self.__angle(self.coordinates[18], self.coordinates[20],
                                             self.coordinates[26])
        index += 1
        # Angle 18
        self.angles[0][index] = self.__angle(self.coordinates[5], self.coordinates[1],
                                             self.coordinates[10])
        index += 1
        # Angle 19
        self.angles[0][index] = self.__angle(self.coordinates[10], self.coordinates[1],
                                             self.coordinates[7])
        index += 1
        # Angle 20
        self.angles[0][index] = self.__angle(self.coordinates[10], self.coordinates[8],
                                             self.coordinates[5])
    '''
    def showGraph(self):

        graph = self.buildGraph(self)

        if (graph is None):
            return

        nodesPositions = networkx.get_node_attributes(graph, "pos")

        for faceEdge in self.edges:
            cv2.line(self.frame_shape, nodesPositions[faceEdge[0]], nodesPositions[faceEdge[1]], (0, 0, 255), 1)

        lenLandMarks = 0
        for faceLandmark in self.indexes:
            lenLandMarks += 1
            landmarkPosition = nodesPositions[faceLandmark]
            cv2.putText(self.frame_shape, str(faceLandmark), landmarkPosition, cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1,
                        cv2.LINE_AA)
            # cv2.circle(image, nodesPositions[faceLandmark], 2, (255,0,0))

            # cv2.putText(image, str(faceLandmark), nodesPositions[faceLandmark], 0, 0.2, (255,0,0))
        print(lenLandMarks);
        cv2.imshow("image", self.frame_shape)


    def buildGraph(self):
        height, width, _ = self.frame_shape.shape
        faceModule = mediapipe.solutions.face_mesh
        processedImage = faceModule.FaceMesh(static_image_mode=True).process(self.frame_shape)
        if (processedImage.multi_face_landmarks is None):
            return

        graph = networkx.Graph()

        # Adds node to the graph
        for faceLandmark in self.indexes:
            landmark = processedImage.multi_face_landmarks[0].landmark[faceLandmark]
            pos = (int(landmark.x * width), int(landmark.y * height))
            graph.add_node(faceLandmark, pos=pos)

        nodesPosition = networkx.get_node_attributes(graph, "pos")
        # Adds edges to the graph

        for faceEdge in self.edges:
            # Calculates the distance

            weight = distance.cityblock(nodesPosition[faceEdge[0]], nodesPosition[faceEdge[1]])

            # If the weight is equal to 0 adds a near null value
            if (weight != 0):
                graph.add_edge(faceEdge[0], faceEdge[1], weight=weight)
            else:
                graph.add_edge(faceEdge[0], faceEdge[1], weight=0.001)

        return graph