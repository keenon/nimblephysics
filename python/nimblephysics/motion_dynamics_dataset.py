import torch
from torch.utils.data import Dataset
import nimblephysics_libs._nimblephysics as nimble
from typing import List, Dict, Callable, Tuple
import numpy as np
import os
import math
import random
import functools


class MotionDynamicsDataset(Dataset):
    """
    This loads a folder full of SubjectOnDisk objects, and then can page in frames as training/test data.

    This is very scalable, because data isn't loaded into RAM until it's needed, and then can be promptly
    freed right after use. That means hundreds of GBs of data is not a problem, even on commodity hardware.

    This assumes that your folder has a Geometry/ folder as a top-level object, and then any number of 
    *.bin files representing SubjectOnDisk's in any level of folder nesting.
    """
    rootDir: str
    geometryFolderPath: str
    subjects: List[nimble.biomechanics.SubjectOnDisk]
    filteredFrames: List[Tuple[int, int, int]]
    numRandomSeeds: int
    featurizeFrame: Callable[[
        nimble.dynamics.Skeleton, nimble.biomechanics.SubjectOnDisk, int, int], Dict[str, torch.Tensor]]
    randomOffset: int

    def __init__(self,
                 rootDir: str,
                 featurizeFrame: Callable[[nimble.dynamics.Skeleton, nimble.biomechanics.SubjectOnDisk, int, int, int], Dict[str, torch.Tensor]],
                 numPassesThroughTheData: int = 1,
                 filterFrames: Callable[[
                     nimble.biomechanics.SubjectOnDisk, int, int], bool] = None,
                 cacheSkeletons: int = 1000,
                 randomOffset: int = random.randint(0, 1000000)):
        self.rootDir = rootDir
        self.geometryFolderPath = os.path.abspath(
            os.path.join(rootDir, './Geometry/')) + '/'
        self.subjects = []
        self.filteredFrames = []
        self.numPassesThroughTheData = numPassesThroughTheData
        self.featurizeFrame = featurizeFrame
        self.randomOffset = randomOffset

        self.getSkeleton = functools.lru_cache(
            maxsize=cacheSkeletons)(self.getSkeleton)

        # r=root, d=directories, f = files
        for root, directories, files in os.walk(rootDir):
            for file in files:
                if file.endswith(".bin"):
                    try:
                        newSubject = nimble.biomechanics.SubjectOnDisk(
                            os.path.join(root, file))
                        subjectIdx = len(self.subjects)
                        self.subjects.append(newSubject)
                        if filterFrames is not None:
                            for trial in range(newSubject.getNumTrials()):
                                for t in range(newSubject.getTrialLength(trial)):
                                    if filterFrames(newSubject, trial, t):
                                        self.filteredFrames.append(
                                            (subjectIdx, trial, t))
                        else:
                            for trial in range(newSubject.getNumTrials()):
                                for t in range(newSubject.getTrialLength(trial)):
                                    self.filteredFrames.append(
                                        (subjectIdx, trial, t))
                    except Exception:
                        # ignore exceptions, this means the file was mangled, the C++ will have printed an error
                        pass

    def __len__(self) -> int:
        return len(self.filteredFrames) * self.numPassesThroughTheData

    def getSkeleton(self, subjectIdx: int):
        return self.subjects[subjectIdx].readSkel(self.geometryFolderPath)

    def __getitem__(self, idx: int):
        index = idx % len(self.filteredFrames)
        subjectIdx, trial, t = self.filteredFrames[index]
        # We add an offset to the random seed, to avoid sharing random seeds between datasets (for example, train and test) which leads to evil bugs
        randomSeed = idx + self.randomOffset
        skel = self.getSkeleton(subjectIdx)
        return self.featurizeFrame(skel, self.subjects[subjectIdx], trial, t, randomSeed)


# Example usage:

# def featurizeFrame(skel: nimble.dynamics.Skeleton, subj: nimble.biomechanics.SubjectOnDisk, trial: int, frame: int, randomSeed: int) -> Dict[str, torch.Tensor]:
#     frames: List[nimble.biomechanics.Frame] = subj.readFrames(trial, frame, 1)
#     return {
#         'test': torch.from_numpy(frames[0].pos)
#     }
# dataset = MotionDynamicsDataset("./testDataset", featurizeFrame)
# print(len(dataset))
# print(dataset[3])
