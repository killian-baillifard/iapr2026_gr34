import numpy as np
from project.scripts.dataset import Player
from project.scripts.dataset import WIDTH as IMAGE_WIDTH, HEIGHT as IMAGE_HEIGHT

SECTOR_WIDTH = 2000
SECTOR_HEIGHT = 1000

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
        x       = IMAGE_WIDTH / 2 - SECTOR_WIDTH / 2,
        y       = IMAGE_HEIGHT / 2 - SECTOR_HEIGHT / 2,
        width   = SECTOR_WIDTH,
        height  = SECTOR_HEIGHT
    ),
    str(Player.P1): Sector(
        x       = IMAGE_WIDTH / 2 - SECTOR_WIDTH / 2,
        y       = IMAGE_HEIGHT - SECTOR_HEIGHT,
        width   = SECTOR_WIDTH,
        height  = SECTOR_HEIGHT
    ),
    str(Player.P2): Sector(
        x       = IMAGE_WIDTH - SECTOR_HEIGHT,
        y       = IMAGE_HEIGHT / 2 - SECTOR_WIDTH / 2,
        width   = SECTOR_HEIGHT,
        height  = SECTOR_WIDTH
    ),
    str(Player.P3): Sector(
        x       = IMAGE_WIDTH / 2 - SECTOR_WIDTH / 2,
        y       = 0,
        width   = SECTOR_WIDTH,
        height  = SECTOR_HEIGHT
    ),
    str(Player.P4): Sector(
        x       = 0,
        y       = IMAGE_HEIGHT / 2 - SECTOR_WIDTH / 2,
        width   = SECTOR_HEIGHT,
        height  = SECTOR_WIDTH
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
        center_sector,
        player_1_sector,
        player_2_sector,
        player_3_sector,
        player_4_sector
    ])
    return sectors
