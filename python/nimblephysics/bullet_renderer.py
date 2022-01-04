import os
from typing import List, Optional, Tuple

import imageio
import nimblephysics as nimble
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc
from scipy.spatial.transform import Rotation
import torch


class BulletRenderer:
    def __init__(self, world: nimble.simulation.World, urdf_paths: List[str]):
        self.world = world

        self._p = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.camera: BulletCamera = BulletCamera(self._p)

        skel_names: List[str] = []
        for skel_id in range(world.getNumSkeletons()):
            skel = world.getSkeleton(skel_id)
            skel_names.append(skel.getName())

        self.bulletBodyDict = {}
        for name, path in zip(skel_names, urdf_paths):
            # Currently, we assume a single rigid body (not articulated) per
            # URDF file that is either free (6 DoFs) or fixed (0 DoFs).
            # For fixed bodies:
            #   A fixed joint between `world` and the body is required in the
            #   URDF file.
            # For free bodies:
            #   Only definition of the single base link is required.
            #   Joint specification is not required because there isn't a way to
            #   specify a free joint in URDF.
            #   The world link is therefore also not required because otherwise
            #   the loader would complain about two root links.
            #   It seems that loading such a URDF in Nimble is treated as a
            #   body attached to the world link via a free joint as well.
            bodyId: int = self._p.loadURDF(path)
            # bodyName: str = self._p.getBodyInfo(bodyUniqueId=bodyId)[1].decode("UTF-8")
            self.bulletBodyDict[name] = bodyId

    def renderStates(
        self,
        states: List[torch.Tensor],
        saveDir: Optional[str] = None,
        frameSkip: Optional[int] = 5,
    ):
        images = []
        for state in states[::frameSkip]:
            self.setState(state)
            image = self.camera.renderState()
            images.append(image)

        if saveDir is not None:
            self.camera.saveImages(images, saveDir)
        return images

    def setState(self, state: torch.Tensor):
        cursor = 0
        for skel_idx in range(self.world.getNumSkeletons()):
            skel = self.world.getSkeleton(skel_idx)
            # For now, only fixed or free joints are supported.
            if skel.getNumDofs() > 0:
                assert skel.getNumDofs() == 6
                skel_state_size = skel.getNumDofs() * 2
                skel_state = state[cursor : cursor + skel_state_size]  # [2D]
                q = skel_state[: skel.getNumDofs()]  # [D]
                self._p.resetBasePositionAndOrientation(
                    bodyUniqueId=self.bulletBodyDict[skel.getName()],
                    posObj=q[3:],
                    ornObj=Rotation.from_rotvec(q[:3]).as_quat(),
                )
                cursor += skel_state_size


class BulletCamera:
    def __init__(self, bc: bc.BulletClient):
        self._p = bc

        self.setViewMatrixParams()
        self.setProjectionMatrixParams()
        self.setImageParameters()

    def renderState(self):
        viewMatrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.targetPosition,
            distance=self.distance,
            yaw=self.yaw,
            pitch=self.pitch,
            roll=self.roll,
            upAxisIndex=self.upAxisIndex,
        )
        projMatrix = self._p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.width / self.height,
            nearVal=self.nearVal,
            farVal=self.farVal,
        )
        lightDirection = [0, 0, 0]
        lightDirection[self.upAxisIndex] = 1
        image = self._p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=viewMatrix,
            projectionMatrix=projMatrix,
            lightDirection=lightDirection,
            renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
        )[2]
        return image

    def saveImages(self, images: List[np.ndarray], saveDir: str):
        saveDir = os.path.join(os.getcwd(), saveDir)
        os.makedirs(saveDir, exist_ok=True)
        for i, image in enumerate(images):
            path = os.path.join(saveDir, f"{i:05}.png")
            imageio.imwrite(path, image)
        gif_path = os.path.join(saveDir, "video.gif")
        imageio.mimsave(gif_path, images, fps=30)
        print(f"Saved images and gif to: {saveDir}")

    def setViewMatrixParams(
        self,
        targetPosition: Tuple[float, float, float] = [0.0, 0.0, 0.0],
        distance: Optional[float] = 1.0,
        yaw: Optional[float] = 0.0,
        pitch: Optional[float] = -30.0,
        roll: Optional[float] = 0.0,
        upAxisIndex: Optional[float] = 1,
    ):
        self.targetPosition = targetPosition
        self.distance = distance

        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

        assert 0 <= upAxisIndex <= 2
        self.upAxisIndex = upAxisIndex

    def setProjectionMatrixParams(
        self,
        fov: Optional[float] = 60.0,
        nearVal: Optional[float] = 0.01,
        farVal: Optional[float] = 100.0,
    ):
        self.fov = fov
        self.nearVal = nearVal
        self.farVal = farVal

    def setImageParameters(
        self, width: Optional[int] = 256, height: Optional[int] = 256
    ):
        self.width = width
        self.height = height
