"""Run a utility to try to get all images from a list of pintrest boards,
adding some randomness into the process to prevent pintrest from blocking
you.
"""
import click
import json
import logging
import pathlib
import pintrest_utils

logger = logging.getLogger('')


@click.command(help=__doc__)
@click.option('-v', '--verbose', count=True)
@click.option('--boards_file', default='boards.json')
@click.option('--boards_dir', default='./boards')
@click.option('--images_file', default='images.json')
@click.option('--download_dir', default='./images')
@click.option('--time_between_bursts', type=float, default=60)
@click.option('--burst_count', type=int, default=8)
def main(verbose, boards_file, boards_dir, images_file, download_dir,
         time_between_bursts, burst_count):
    logging.basicConfig(level=logging.INFO if not verbose else logging.DEBUG)
    logger.info(f"Starting scraper with settings:\nboards_file: {boards_file}"
                f"\nimages_file: {images_file}\ndownload_dir: {download_dir}")

    logger.debug("Loading board file")
    with open(boards_file, 'r') as fh:
        pin_boards = json.load(fh)

    for board in pin_boards:
        logger.debug(f"Processing {board}")

        board_pins = pintrest_utils.fetch_board_rss(board,
                                                    cache_dir=boards_dir)
        logger.debug(f"Got {len(board_pins)} from board")

        pintrest_utils.download_all_pins_at_random_interval(
            board_pins, time_between_bursts, burst_count)


if __name__ == "__main__":
    main()
