"""Tests for the SlimProto CLI."""

import asyncio
from itertools import chain
import logging
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock
from urllib.parse import quote

import pytest

from aioslimproto.cli import CometDClient, SlimProtoCLI, create_player_item
from aioslimproto.client import SlimClient
from aioslimproto.models import EventType, PlayerState, SlimEvent
from aioslimproto.server import SlimServer


@pytest.fixture
def writer() -> Mock:
    """Create a mocked CLI connection writer."""
    result = Mock()
    result.drain = AsyncMock()
    result.is_closing.return_value = False
    result.wait_closed = AsyncMock()
    return result


async def send_cli_command(
    cli: SlimProtoCLI,
    writer: Mock,
    command: str,
) -> bytes:
    """Send one complete command through the in-memory CLI connection."""
    reader = asyncio.StreamReader()
    reader.feed_data(f"{command}\n".encode())
    reader.feed_eof()

    await cli._handle_cli_client(reader, writer)  # noqa: SLF001 # Accessing a protected member for testing purposes

    assert writer.write.call_count == 1
    return writer.write.call_args.args[0]


def encode_response(*tags: str | dict[str, str | int]) -> bytes:
    """Encode CLI response tags as the legacy CLI does."""
    flat_tags = chain.from_iterable(
        (tag,)
        if isinstance(tag, str)
        else (f"{key}:{value}" for key, value in tag.items())
        for tag in tags
    )
    return f"{' '.join(quote(tag) for tag in flat_tags)}\n".encode("iso-8859-1")


@pytest.fixture
def dummy_player() -> SlimClient:
    """Create a dummy player for testing."""
    return cast(
        "SlimClient",
        SimpleNamespace(
            player_id="a5:41:d2:cd:cd:05",
            name="Kitchen",
            device_model="Squeezebox Touch",
            connected=True,
            state=PlayerState.STOPPED,
            powered=True,
            device_type="touch",
            extra_data={"uuid": "player-uuid-1", "seq_no": 1},
            device_address="127.0.0.1",
            current_media=None,
            next_media=None,
            elapsed_seconds=0,
            volume_level=50,
            volume_set=AsyncMock(),
            muted=False,
            mute=AsyncMock(),
            presets=[],
        ),
    )


@pytest.fixture
def dummy_server(dummy_player: SlimClient) -> SlimServer:
    """Create a dummy server for testing."""
    return cast(
        "SlimServer",
        SimpleNamespace(
            name="testserver",
            logger=logging.getLogger(),
            players=[dummy_player],
            get_player=lambda player_id: (
                dummy_player if player_id == "a5:41:d2:cd:cd:05" else None
            ),
        ),
    )


class TestPlayersCommand:
    """Tests for the players command.

    Reference: https://lyrion.org/reference/cli/players/#players
    """

    @pytest.fixture
    def several_dummy_players(self, dummy_player: SlimClient) -> list[SlimClient]:
        """Create several dummy players for testing."""
        return [
            dummy_player,
            cast(
                "SlimClient",
                SimpleNamespace(
                    player_id="player-2",
                    name="Living Room",
                    device_model="Squeezebox Radio",
                    connected=True,
                    state=PlayerState.PLAYING,
                    powered=True,
                    device_type="radio",
                    extra_data={"uuid": "player-uuid-2", "seq_no": 2},
                    device_address="127.0.0.2",
                ),
            ),
            cast(
                "SlimClient",
                SimpleNamespace(
                    player_id="player-3",
                    name="Bedroom",
                    device_model="Squeezebox Controller",
                    connected=True,
                    state=PlayerState.PLAYING,
                    powered=True,
                    device_type="controller",
                    extra_data={"uuid": "player-uuid-3", "seq_no": 3},
                    device_address="127.0.0.3",
                ),
            ),
        ]

    @pytest.mark.asyncio
    async def test_returns_all_registered_players(
        self, writer: Mock, several_dummy_players: list[SlimClient]
    ) -> None:
        """Should return all players when no limit is specified."""
        server = cast(
            "SlimServer",
            SimpleNamespace(logger=logging.getLogger(), players=several_dummy_players),
        )

        cli = SlimProtoCLI(server)

        response = await send_cli_command(cli, writer, "players 0")
        expected_response = encode_response(
            "players",
            "0",
            "count:3",
            {
                "playerindex": 0,
                "playerid": "a5:41:d2:cd:cd:05",
                "name": "Kitchen",
                "modelname": "Squeezebox Touch",
                "connected": 1,
                "isplaying": 0,
                "power": 1,
                "model": "touch",
                "canpoweroff": 1,
                "firmware": "unknown",
                "isplayer": 1,
                "displaytype": "none",
                "uuid": "player-uuid-1",
                "seq_no": 1,
                "ip": "127.0.0.1",
            },
            {
                "playerindex": 1,
                "playerid": "player-2",
                "name": "Living Room",
                "modelname": "Squeezebox Radio",
                "connected": 1,
                "isplaying": 1,
                "power": 1,
                "model": "radio",
                "canpoweroff": 1,
                "firmware": "unknown",
                "isplayer": 1,
                "displaytype": "none",
                "uuid": "player-uuid-2",
                "seq_no": 2,
                "ip": "127.0.0.2",
            },
            {
                "playerindex": 2,
                "playerid": "player-3",
                "name": "Bedroom",
                "modelname": "Squeezebox Controller",
                "connected": 1,
                "isplaying": 1,
                "power": 1,
                "model": "controller",
                "canpoweroff": 1,
                "firmware": "unknown",
                "isplayer": 1,
                "displaytype": "none",
                "uuid": "player-uuid-3",
                "seq_no": 3,
                "ip": "127.0.0.3",
            },
        )

        assert response == expected_response

    @pytest.mark.asyncio
    async def test_returns_second_player(
        self, writer: Mock, several_dummy_players: list[SlimClient]
    ) -> None:
        """Should return the 2nd player when offset and limit are set to 1."""
        server = cast(
            "SlimServer",
            SimpleNamespace(logger=logging.getLogger(), players=several_dummy_players),
        )

        cli = SlimProtoCLI(server)

        response = await send_cli_command(cli, writer, "players 1 1")
        expected_response = encode_response(
            "players",
            "1",
            "1",
            "count:1",
            {
                "playerindex": 1,
                "playerid": "player-2",
                "name": "Living Room",
                "modelname": "Squeezebox Radio",
                "connected": 1,
                "isplaying": 1,
                "power": 1,
                "model": "radio",
                "canpoweroff": 1,
                "firmware": "unknown",
                "isplayer": 1,
                "displaytype": "none",
                "uuid": "player-uuid-2",
                "seq_no": 2,
                "ip": "127.0.0.2",
            },
        )

        assert response == expected_response


class TestStatusCommand:
    """Tests for the status command.

    Reference: https://lyrion.org/reference/cli/compoundqueries/#status
    """

    @pytest.mark.asyncio
    async def test_echoes_without_player(
        self, writer: Mock, dummy_server: SlimServer
    ) -> None:
        """Should echo the command when no player is specified."""
        cli = SlimProtoCLI(dummy_server)

        response = await send_cli_command(cli, writer, "status 0 2 tags:")
        expected_response = encode_response("status", "0", "2", "tags:")

        assert response == expected_response

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Many values are hard-coded")
    async def test_simple_example(self, writer: Mock, dummy_server: SlimServer) -> None:
        """The "simple example" on the reference page should work."""
        cli = SlimProtoCLI(dummy_server)

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 status 0 2 tags:"
        )
        expected_response = encode_response(
            "a5:41:d2:cd:cd:05",
            "status",
            "0",
            "2",
            "tags:",
            {
                "player_name": "Kitchen",
                "player_connected": 1,
                "player_needs_upgrade": 0,
                "player_is_upgrading": 0,
                "power": 1,
                "signalstrength": 50,
                "waitingToPlay": 0,
                "mode": "stop",
                "remote": 1,
                "current_title": "testserver",
                "time": 0,
                "duration": 0,
                "mixer volume": 50,
                # Not documented, but sent by the server (https://github.com/LMS-Community/slimserver/tree/cf756254749c489a1ac859dd4aad139b513dc655/Slim/Control/Queries.pm#L4055)
                "player_ip": "127.0.0.1",
                "playlist_cur_index": 1,
                "playlist_tracks": 0,
                "uuid": "player-uuid-1",
                "seq_no": 1,
            },
        )

        assert response == expected_response


class TestMixerVolumeCommand:
    """Tests for the mixer volume command.

    Reference: https://lyrion.org/reference/cli/players/#mixer-volume
    """

    @pytest.mark.asyncio
    async def test_get_volume(self, writer: Mock, dummy_player: SlimClient) -> None:
        """Should return the volume when requested with a '?'."""
        server = cast(
            "SlimServer",
            SimpleNamespace(
                logger=logging.getLogger(),
                players=[dummy_player],
                get_player=lambda player_id: (
                    dummy_player if player_id == "a5:41:d2:cd:cd:05" else None
                ),
            ),
        )

        cli = SlimProtoCLI(server)

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer volume ?"
        )
        expected_response = encode_response(
            "a5:41:d2:cd:cd:05", "mixer", "volume", "50"
        )

        assert response == expected_response

    @pytest.mark.asyncio
    async def test_set_volume_absolute(
        self, writer: Mock, dummy_player: SlimClient
    ) -> None:
        """Should set the volume to an absolute value."""
        server = cast(
            "SlimServer",
            SimpleNamespace(
                logger=logging.getLogger(),
                players=[dummy_player],
                get_player=lambda player_id: (
                    dummy_player if player_id == "a5:41:d2:cd:cd:05" else None
                ),
            ),
        )

        cli = SlimProtoCLI(server)

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer volume 75"
        )
        expected_response = encode_response(
            "a5:41:d2:cd:cd:05", "mixer", "volume", "75"
        )

        assert response == expected_response
        cast("Mock", dummy_player.volume_set).assert_called_once_with(75)

    @pytest.mark.asyncio
    async def test_set_volume_relative(
        self, writer: Mock, dummy_player: SlimClient
    ) -> None:
        """Should change the volume by a relative value."""
        server = cast(
            "SlimServer",
            SimpleNamespace(
                logger=logging.getLogger(),
                players=[dummy_player],
                get_player=lambda player_id: (
                    dummy_player if player_id == "a5:41:d2:cd:cd:05" else None
                ),
            ),
        )

        cli = SlimProtoCLI(server)

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer volume +25"
        )
        expected_response = encode_response(
            "a5:41:d2:cd:cd:05", "mixer", "volume", "+25"
        )

        assert response == expected_response
        cast("Mock", dummy_player.volume_set).assert_called_once_with(75)

        cast("Mock", dummy_player.volume_set).reset_mock()
        writer.reset_mock()

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer volume -50"
        )
        expected_response = encode_response(
            "a5:41:d2:cd:cd:05", "mixer", "volume", "-50"
        )

        assert response == expected_response
        cast("Mock", dummy_player.volume_set).assert_called_once_with(0)

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Fractional volume is not yet implemented")
    async def test_set_volume_fractional(
        self, writer: Mock, dummy_player: SlimClient
    ) -> None:
        """Should set the volume to a fractional value."""
        server = cast(
            "SlimServer",
            SimpleNamespace(
                logger=logging.getLogger(),
                players=[dummy_player],
                get_player=lambda player_id: (
                    dummy_player if player_id == "a5:41:d2:cd:cd:05" else None
                ),
            ),
        )

        cli = SlimProtoCLI(server)

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer volume 25.5"
        )
        expected_response = encode_response(
            "a5:41:d2:cd:cd:05", "mixer", "volume", "25.5"
        )

        assert response == expected_response
        cast("Mock", dummy_player.volume_set).assert_called_once_with(25.5)


class TestMixerMutingCommand:
    """Tests for the mixer muting command.

    Reference: https://lyrion.org/reference/cli/players/#mixer-muting
    """

    @pytest.mark.asyncio
    async def test_can_toggle_muting(
        self, writer: Mock, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """Should allow toggling the muting."""
        cli = SlimProtoCLI(dummy_server)

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer muting toggle"
        )
        expected_response = encode_response(
            "a5:41:d2:cd:cd:05", "mixer", "muting", "toggle"
        )

        assert response == expected_response
        cast("Mock", dummy_player.mute).assert_called_once_with(True)  # noqa: FBT003 # This is how it's called in the original code

    @pytest.mark.asyncio
    async def test_can_toggle_muting_without_keyword(
        self, writer: Mock, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """Should allow toggling the muting without the 'toggle' keyword."""
        cli = SlimProtoCLI(dummy_server)

        response = await send_cli_command(cli, writer, "a5:41:d2:cd:cd:05 mixer muting")
        expected_response = encode_response("a5:41:d2:cd:cd:05", "mixer", "muting")

        assert response == expected_response
        cast("Mock", dummy_player.mute).assert_called_once_with(True)  # noqa: FBT003 # This is how it's called in the original code

    @pytest.mark.asyncio
    async def test_can_mute(
        self, writer: Mock, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """Should allow muting regardless of the current state."""
        cli = SlimProtoCLI(dummy_server)

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer muting 1"
        )
        expected_response = encode_response("a5:41:d2:cd:cd:05", "mixer", "muting", "1")

        assert response == expected_response
        cast("Mock", dummy_player.mute).assert_called_once_with(True)  # noqa: FBT003 # This is how it's called in the original code

    @pytest.mark.asyncio
    async def test_can_unmute(
        self, writer: Mock, dummy_server: SlimServer, dummy_player: SlimClient
    ) -> None:
        """Should allow unmuting regardless of the current state."""
        cli = SlimProtoCLI(dummy_server)

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer muting 0"
        )
        expected_response = encode_response("a5:41:d2:cd:cd:05", "mixer", "muting", "0")

        assert response == expected_response
        cast("Mock", dummy_player.mute).assert_called_once_with(False)  # noqa: FBT003 # This is how it's called in the original code

    @pytest.mark.asyncio
    async def test_can_query_muting(
        self, writer: Mock, dummy_server: SlimServer
    ) -> None:
        """Should return the current state of muting."""
        cli = SlimProtoCLI(dummy_server)

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer muting ?"
        )
        expected_response = encode_response("a5:41:d2:cd:cd:05", "mixer", "muting", "0")
        assert response == expected_response


class TestCommandHandler:
    """Tests for application-provided Slim command handling."""

    @pytest.mark.asyncio
    async def test_replaces_each_query_marker_with_scalar_response(
        self, writer: Mock, dummy_server: SlimServer
    ) -> None:
        """Should pass parsed commands to the handler and substitute query values."""
        handler = AsyncMock(return_value=[12, 34])
        cli = SlimProtoCLI(dummy_server, command_handler=handler)

        response = await send_cli_command(
            cli,
            writer,
            "a5:41:d2:cd:cd:05 info total genres ? ? tags:summary offset:2",
        )

        assert response == encode_response(
            "a5:41:d2:cd:cd:05",
            "info",
            "total",
            "genres",
            "12",
            "34",
            "tags:summary",
            "offset:2",
        )

    @pytest.mark.asyncio
    async def test_returns_complex_response_blocks(
        self, writer: Mock, dummy_server: SlimServer
    ) -> None:
        """Should append each complex response block to the echoed query."""
        cli = SlimProtoCLI(
            dummy_server,
            command_handler=AsyncMock(
                return_value=[
                    {"id": 1, "name": "First"},
                    {"id": 2, "name": "Second"},
                ]
            ),
        )

        response = await send_cli_command(cli, writer, "library items")

        assert response == encode_response(
            "library",
            "items",
            {"id": 1, "name": "First"},
            {"id": 2, "name": "Second"},
        )

    @pytest.mark.asyncio
    async def test_none_response_echoes_the_request(
        self, writer: Mock, dummy_server: SlimServer
    ) -> None:
        """Should echo a command handled without a response payload."""
        cli = SlimProtoCLI(dummy_server, command_handler=AsyncMock(return_value=None))

        response = await send_cli_command(cli, writer, "external command value")

        assert response == encode_response("external", "command", "value")

    @pytest.mark.asyncio
    async def test_not_implemented_falls_back_to_builtin_handler(
        self, writer: Mock, dummy_server: SlimServer
    ) -> None:
        """Should use the built-in handler when the application declines a command."""
        cli = SlimProtoCLI(
            dummy_server, command_handler=AsyncMock(side_effect=NotImplementedError)
        )

        response = await send_cli_command(
            cli, writer, "a5:41:d2:cd:cd:05 mixer volume ?"
        )

        assert response == encode_response("a5:41:d2:cd:cd:05", "mixer", "volume", "50")

    @pytest.mark.asyncio
    async def test_menustatus_returns_full_home_menu(
        self, dummy_server: SlimServer
    ) -> None:
        """A menustatus request answers with the home menu and the 'add' directive."""
        menu_items = [{"id": "ma_ha_scripts", "text": "HA scripts"}]
        cli = SlimProtoCLI(
            dummy_server,
            command_handler=AsyncMock(return_value={"item_loop": menu_items}),
        )

        result = await cli._handle_menustatus("a5:41:d2:cd:cd:05")  # noqa: SLF001

        assert result == ["a5:41:d2:cd:cd:05", menu_items, "add", "a5:41:d2:cd:cd:05"]

    @pytest.mark.asyncio
    async def test_menustatus_without_menu_items_returns_none(
        self, dummy_server: SlimServer
    ) -> None:
        """An empty home menu yields no menustatus payload."""
        cli = SlimProtoCLI(
            dummy_server,
            command_handler=AsyncMock(return_value={"item_loop": []}),
        )

        assert await cli._handle_menustatus("a5:41:d2:cd:cd:05") is None  # noqa: SLF001


class TestPlayerDisplayName:
    """The server-reported player name prefers the configured display name."""

    def test_create_player_item_prefers_display_name(
        self, dummy_player: SlimClient
    ) -> None:
        """A configured display name is used for the players list."""
        dummy_player.display_name = "Küchen Radio"

        assert create_player_item(0, dummy_player)["name"] == "Küchen Radio"

    def test_create_player_item_falls_back_to_device_name(
        self, dummy_player: SlimClient
    ) -> None:
        """Without a display name the device-reported name is used."""
        dummy_player.display_name = None

        assert create_player_item(0, dummy_player)["name"] == "Kitchen"

    @pytest.mark.asyncio
    async def test_status_reports_display_name(
        self, dummy_player: SlimClient, dummy_server: SlimServer
    ) -> None:
        """The playerstatus name reflects the configured display name."""
        dummy_player.display_name = "Küchen Radio"
        cli = SlimProtoCLI(dummy_server)

        result = await cli._handle_status(dummy_player.player_id)  # noqa: SLF001

        assert result["player_name"] == "Küchen Radio"


class TestCometdLongPolling:
    """The CometD long-poll wait must not hold up non-connect meta requests."""

    @pytest.mark.asyncio
    async def test_handshake_is_answered_immediately(
        self, dummy_server: SlimServer
    ) -> None:
        """A handshake must not be held open for the long-poll timeout."""
        cli = SlimProtoCLI(dummy_server)
        request = Mock()
        request.json = AsyncMock(
            return_value=[
                {
                    "channel": "/meta/handshake",
                    "id": "1",
                    "version": "1.0",
                    "supportedConnectionTypes": ["streaming"],
                }
            ]
        )

        response = await asyncio.wait_for(
            cli._handle_cometd_client(request),  # noqa: SLF001
            timeout=2,
        )

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_long_poll_connect_returns_queued_event(
        self, dummy_server: SlimServer
    ) -> None:
        """A connect long-poll still waits for and returns a queued event."""
        cli = SlimProtoCLI(dummy_server)
        handshake = Mock()
        handshake.json = AsyncMock(
            return_value=[
                {
                    "channel": "/meta/handshake",
                    "id": "1",
                    "version": "1.0",
                    "supportedConnectionTypes": ["streaming"],
                }
            ]
        )
        await cli._handle_cometd_client(handshake)  # noqa: SLF001
        clientid = next(iter(cli._cometd_clients))  # noqa: SLF001
        cli._cometd_clients[clientid].queue.put_nowait(  # noqa: SLF001
            {"channel": "/slim/test", "data": "event"}
        )
        request = Mock()
        request.json = AsyncMock(
            return_value=[
                {
                    "channel": "/meta/connect",
                    "id": "2",
                    "clientId": clientid,
                    "connectionType": "long-polling",
                }
            ]
        )

        response = await asyncio.wait_for(
            cli._handle_cometd_client(request),  # noqa: SLF001
            timeout=2,
        )

        assert response.status == 200


class TestPlaylistHandler:
    """The status menu can be served from an application-provided playlist page."""

    @pytest.mark.asyncio
    async def test_status_menu_uses_playlist_handler(
        self, dummy_server: SlimServer
    ) -> None:
        """The status menu mode merges the playlist page from the handler."""
        handler = AsyncMock(
            return_value={
                "count": 42,
                "offset": 0,
                "playlist_tracks": 42,
                "playlist_cur_index": 3,
                "item_loop": [
                    {"text": "Track 1", "actions": {"go": {"cmd": ["playlist"]}}}
                ],
            }
        )
        cli = SlimProtoCLI(dummy_server, playlist_handler=handler)

        result = await cli._handle_status("a5:41:d2:cd:cd:05", 0, 200, menu="menu")  # noqa: SLF001

        assert result["count"] == 42
        assert result["playlist_tracks"] == 42
        assert result["playlist_cur_index"] == 3
        assert result["item_loop"][0]["text"] == "Track 1"
        handler.assert_awaited_once_with("a5:41:d2:cd:cd:05", 0, 200)

    @pytest.mark.asyncio
    async def test_status_menu_falls_back_without_handler(
        self, dummy_server: SlimServer
    ) -> None:
        """Without a playlist handler the status menu keeps its built-in item_loop."""
        cli = SlimProtoCLI(dummy_server)

        result = await cli._handle_status("a5:41:d2:cd:cd:05", 0, 200, menu="menu")  # noqa: SLF001

        assert result["count"] == 0
        assert result["item_loop"] == []


class TestStatusBaseMore:
    """The status menu 'more' action differs between Now Playing and the playlist."""

    @pytest.mark.asyncio
    async def test_now_playing_more_opens_the_playlist(
        self, dummy_server: SlimServer
    ) -> None:
        """On the now-playing screen, 'more' opens the queue."""
        cli = SlimProtoCLI(dummy_server)

        result = await cli._handle_status("a5:41:d2:cd:cd:05", "-", 10, menu="menu")  # noqa: SLF001

        more = result["base"]["actions"]["more"]
        assert more["nextWindow"] == "playlist"
        assert "window" not in more

    @pytest.mark.asyncio
    async def test_playlist_more_is_the_context_menu(
        self, dummy_server: SlimServer
    ) -> None:
        """In the playlist window, 'more' stays the per-row context menu."""
        cli = SlimProtoCLI(dummy_server)

        result = await cli._handle_status("a5:41:d2:cd:cd:05", 0, 200, menu="menu")  # noqa: SLF001

        more = result["base"]["actions"]["more"]
        assert more["cmd"] == ["contextmenu"]
        assert more["window"] == {"isContextMenu": 1}


class TestPowerEventSubscriptions:
    """A power change re-pushes the device's own status/displaystatus queries."""

    @staticmethod
    def _cometd_client() -> CometDClient:
        """Build a CometD client subscribed to several player queries."""
        client = CometDClient(client_id="cid", player_id="a5:41:d2:cd:cd:05")
        client.slim_subscriptions["/slim/status"] = {
            "id": 1,
            "data": {
                "response": "/slim/status",
                "request": ["a5:41:d2:cd:cd:05", ["status", "-", 10]],
            },
        }
        client.slim_subscriptions["/slim/displaystatus"] = {
            "id": 2,
            "data": {
                "response": "/slim/displaystatus",
                "request": [
                    "a5:41:d2:cd:cd:05",
                    ["displaystatus", "subscribe:showbriefly"],
                ],
            },
        }
        client.slim_subscriptions["/slim/serverstatus"] = {
            "id": 3,
            "data": {
                "response": "/slim/serverstatus",
                "request": ["", ["serverstatus", 0, 50]],
            },
        }
        return client

    @staticmethod
    def _pushed_responses(cli: SlimProtoCLI) -> set[str]:
        """Return the response channels the CLI re-sent during the event."""
        return {
            call.args[1]["data"]["response"]
            for call in cli._handle_cometd_client_request.call_args_list  # noqa: SLF001
        }

    @pytest.mark.asyncio
    async def test_power_event_pushes_status_and_displaystatus(
        self, dummy_server: SlimServer
    ) -> None:
        """The device learns its new power from the pushed status query."""
        cli = SlimProtoCLI(dummy_server)
        cli._cometd_clients["cid"] = self._cometd_client()  # noqa: SLF001
        cli._handle_cometd_client_request = Mock()  # noqa: SLF001

        await cli._on_player_event(  # noqa: SLF001
            SlimEvent(EventType.PLAYER_POWER_UPDATED, "a5:41:d2:cd:cd:05")
        )

        pushed = self._pushed_responses(cli)
        assert "/slim/status" in pushed
        assert "/slim/displaystatus" in pushed

    @pytest.mark.asyncio
    async def test_plain_update_does_not_push_status(
        self, dummy_server: SlimServer
    ) -> None:
        """A regular update must not flood the device with a full status push."""
        cli = SlimProtoCLI(dummy_server)
        cli._cometd_clients["cid"] = self._cometd_client()  # noqa: SLF001
        cli._handle_cometd_client_request = Mock()  # noqa: SLF001

        await cli._on_player_event(  # noqa: SLF001
            SlimEvent(EventType.PLAYER_UPDATED, "a5:41:d2:cd:cd:05")
        )

        pushed = self._pushed_responses(cli)
        assert "/slim/status" not in pushed
        assert "/slim/displaystatus" not in pushed
