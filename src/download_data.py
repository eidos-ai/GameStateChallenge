from SoccerNet.Downloader import SoccerNetDownloader  # type: ignore

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="data/SoccerNetGS")
mySoccerNetDownloader.downloadDataTask(
    task="gamestate-2024", split=["train", "valid", "test", "challenge"]
)
