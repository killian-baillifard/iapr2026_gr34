import numpy as np
import cv2
from scripts.dataset import Player
from scripts.dataset import WIDTH as IMAGE_WIDTH, HEIGHT as IMAGE_HEIGHT

SECTOR_SRC_WIDTH = 2048
SECTOR_SRC_HEIGHT = 1024

SECTOR_END_WIDTH = 512
SECTOR_END_HEIGHT = 256
SECTOR_END_SIZE = (SECTOR_END_WIDTH, SECTOR_END_HEIGHT)

class Sector:

    def __init__(self, x: int, y: int, width: int, height: int) -> None:

        # Assert sector is contained in image bounds
        assert 0 <= x and x <= IMAGE_WIDTH
        assert 0 <= y and y <= IMAGE_HEIGHT
        assert 0 <= x + width and x + width <= IMAGE_WIDTH
        assert 0 <= y + height and y + height <= IMAGE_HEIGHT

        # Save sector dimensions
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)

    def slice(self, image: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------

        image : np.ndarray
            RGB image of whole table (height, width, channels)
        
        Returns
        -------

        sectors : np.ndarray 
            RGB image of this sector (sector, sector_height, sector_width, channels)
        """
        return image[self.y:(self.y + self.height), self.x:(self.x + self.width), :]

SECTORS = {
    "center": Sector(
        x       = IMAGE_WIDTH / 2 - SECTOR_SRC_WIDTH / 2,
        y       = IMAGE_HEIGHT / 2 - SECTOR_SRC_HEIGHT / 2,
        width   = SECTOR_SRC_WIDTH,
        height  = SECTOR_SRC_HEIGHT
    ),
    str(Player.P1): Sector(
        x       = IMAGE_WIDTH / 2 - SECTOR_SRC_WIDTH / 2,
        y       = IMAGE_HEIGHT - SECTOR_SRC_HEIGHT,
        width   = SECTOR_SRC_WIDTH,
        height  = SECTOR_SRC_HEIGHT
    ),
    str(Player.P2): Sector(
        x       = IMAGE_WIDTH - SECTOR_SRC_HEIGHT,
        y       = IMAGE_HEIGHT / 2 - SECTOR_SRC_WIDTH / 2,
        width   = SECTOR_SRC_HEIGHT,
        height  = SECTOR_SRC_WIDTH
    ),
    str(Player.P3): Sector(
        x       = IMAGE_WIDTH / 2 - SECTOR_SRC_WIDTH / 2,
        y       = 0,
        width   = SECTOR_SRC_WIDTH,
        height  = SECTOR_SRC_HEIGHT
    ),
    str(Player.P4): Sector(
        x       = 0,
        y       = IMAGE_HEIGHT / 2 - SECTOR_SRC_WIDTH / 2,
        width   = SECTOR_SRC_HEIGHT,
        height  = SECTOR_SRC_WIDTH
    )
}

def slice_sectors(image: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------

    image : np.ndarray
        Whole RGB image (height, width, 3)
    
    Returns
    -------

    sectors : np.ndarray
        RGB image sliced in sectors (5, height, width, 3)
    """

    # Slice each sector
    center_sector = SECTORS["center"].slice(image)
    player_1_sector = SECTORS[str(Player.P1)].slice(image)
    player_2_sector = SECTORS[str(Player.P2)].slice(image)
    player_3_sector = SECTORS[str(Player.P3)].slice(image)
    player_4_sector = SECTORS[str(Player.P4)].slice(image)

    # Rotate players 2 to 4 sectors
    player_2_sector = np.rot90(player_2_sector, k=-1)
    player_3_sector = np.rot90(player_3_sector, k=2)
    player_4_sector = np.rot90(player_4_sector, k=1)

    # Return stacked sectors
    sectors = np.stack([
        cv2.resize(center_sector, SECTOR_END_SIZE, interpolation=cv2.INTER_AREA),
        cv2.resize(player_1_sector, SECTOR_END_SIZE, interpolation=cv2.INTER_AREA),
        cv2.resize(player_2_sector, SECTOR_END_SIZE, interpolation=cv2.INTER_AREA),
        cv2.resize(player_3_sector, SECTOR_END_SIZE, interpolation=cv2.INTER_AREA),
        cv2.resize(player_4_sector, SECTOR_END_SIZE, interpolation=cv2.INTER_AREA)
    ])

    return sectors
