__author__ = "Felix"

import asyncio
import csv
import json
import math
import os
import random
import re
import uuid
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Optional, Dict, Any, Set, List, Tuple
import logging

import aiohttp
import aiopg
import core
import discord
from discord.ext import commands, tasks

# Constants
BYPASS_LIST = [
    767824073186869279,  # abbi
    249568050951487499,  # akhil
    323473569008975872,  # olly
    381170131721781248,  # crois
    346382745817055242,  # felix
    211368856839520257,  # illy
    335415340190269440,  # Olly's alt
]

ROLE_HIERARCHY = [
    "1248340570275971125",
    "1248340594686820403",
    "1248340609727729795",
    "1248340626773381240",
    "1248340641117765683",
]

THUMBNAIL = (
    "https://cdn.discordapp.com/attachments/1208495821868245012/1291896171555455027/CleanShot_2024-10-04_"
    "at_23.53.582x.png?ex=6701c391&is=67007211&hm=1138ae2d92387ecde7be34af238bd756462970de2ca6ca559c6aa091f9"
    "32a8ae&"
)
FOOTER = "Sponsored by the Guides Committee"

gamepasses = {
    "Rainbow Name": 20855496,
    "Ground Crew": 20976711,
    "Cabin Crew": 20976738,
    "Captain": 20976820,
    "Senior Staff": 20976845,
    "Staff Manager": 20976883,
    "Airport Manager": 20976943,
    "Board of Directors": 21002253,
    "Co Owner": 21002275,
    "First Class": 21006608,
    "Segway Board": 22042259,
}

colours = {
    "green": 0xA9DC76,
    "red": 0xFF6188,
    "yellow": 0xFFD866,
    "light_blue": 0x78DCE8,
    "purple": 0xAB9DF2,
}

channel_options = {
    "Main": "1221162035513917480",
    "Prize Claims": "1213885924581048380",
    "Affiliate": "1196076391242944693",
    "Development": "1196076499137200149",
    "Appeals": "1196076578938036255",
    "Moderator Reports": "1246863733355970612",
}

UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
EMOJI_VALUES = {True: "✅", False: "⛔"}
K_VALUE = 0.099

BLOXLINK_API_KEY = os.environ.get("BLOXLINK_KEY")
PASSWORD = os.environ.get("POSTGRES_PASSW")
SERVER_ID = "788228600079843338"
HEADERS = {"Authorization": BLOXLINK_API_KEY}

dsn = f"dbname=tickets user=cityairways password={PASSWORD} host=citypostgres"

MOTIVATIONAL_QUOTES = [
    "To toil unyielding is to defy the heavens and earn thy rightful glory.",
    "The weak are but shadows, while the strong etch their names upon stone.",
    "He who fears the wrath of kings shall forever dwell in servitude.",
    "Deny not thy ambitions; for he who grovels shall inherit naught but dust.",
    "Only the bold dare sip from the cup of destiny.",
    "Mercy is oft a veil for cowardice; strike while the iron is hot.",
    "In the forge of adversity, only the resolute are deemed worthy.",
    "The gods aid those who carve their own path with blade and wit.",
    "Hesitation is the dirge of the unworthy; act, or be forgotten.",
    "A throne is never granted; it is seized by the audacious.",
    "The stars weep not for those who falter; rise, or perish in obscurity.",
    "Lo, the meek may inherit the earth, but the strong claim the heavens.",
    "Idleness is the herald of decay; only the industrious shall thrive.",
    "Pity not the fallen; they are but stepping stones for the ambitious.",
    "A man's worth is measured by the foes he dares to challenge.",
    "The silence of the oppressed is the triumph of tyranny; speak, or be chained.",
    "In the arena of life, the lion devours the lamb; be no lamb.",
    "The fates weave for the daring, not for the docile.",
    "To conquer is not cruel; it is the order of the strong over the weak.",
    "Dreams unpursued are but whispers in the wind, meaningless and fleeting.",
    "Shatter thy chains, or be content to rot in servitude.",
    "Gold favors the cunning, not the pious or the hesitant.",
    "A kingdom's glory is built upon the ashes of the defeated.",
    "Suffer not the mediocrity of others to chain thy spirit.",
    "The edge of a sword speaks louder than a hundred pleas.",
    "Seek not the approval of others, for it is a prison for the ambitious.",
    "The path to greatness is paved with the bones of thy failures.",
    "Turn not thy cheek to insults; forbearance breeds contempt.",
    "To reign is to wield power; to follow is to endure obscurity.",
    "The wise sow discord among their rivals to reap unity for themselves.",
]

def find_most_similar(name: str) -> Tuple[str, int]:
    """Find the most similar gamepass name"""
    return max(gamepasses.items(), key=lambda x: SequenceMatcher(None, x[0], name).ratio())


class AsyncCooldownMapping:
    """Handles cooldown calculations asynchronously"""

    def __init__(self, pool, k_value=0.099):
        self.pool = pool
        self.k_value = k_value
        self._cache = {}

    async def get_tickets_and_cooldown(self, user_id: int) -> Tuple[int, float]:
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT COUNT(*) 
                    FROM tickets 
                    WHERE user_id = %s
                    AND DATE_TRUNC('week', timestamp) = DATE_TRUNC('week', CURRENT_DATE)
                """, (user_id,))
                tickets = (await cur.fetchone())[0]

        if tickets < 5:
            return tickets, 0
        if 5 <= tickets < 36.6:
            time = math.exp(self.k_value * tickets)
        else:
            time = math.exp(self.k_value * 36.6)

        return tickets, time * 60

    async def get_cooldown(self, ctx: commands.Context) -> Optional[commands.Cooldown]:
        if ctx.author.id in BYPASS_LIST:
            return None

        tickets, cooldown = await self.get_tickets_and_cooldown(ctx.author.id)
        if cooldown is None:
            return None

        return commands.Cooldown(1, cooldown)


class DatabaseManager:
    """Manages all database operations"""

    def __init__(self, pool):
        self.pool = pool

    async def count_user_tickets(self, user_id: int, period: str) -> int:
        query_map = {
            'day': "DATE(timestamp) = CURRENT_DATE",
            'week': "DATE_TRUNC('week', timestamp) = DATE_TRUNC('week', CURRENT_DATE)",
            'month': "DATE_TRUNC('month', timestamp) = DATE_TRUNC('month', CURRENT_DATE)"
        }

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    SELECT COUNT(*) 
                    FROM tickets 
                    WHERE user_id = %s
                    AND {query_map[period]};
                """, (user_id,))
                return (await cur.fetchone())[0]

    async def add_ticket(self, user_id: int) -> None:
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO tickets (user_id) VALUES (%s);",
                    (user_id,)
                )

    async def get_tickets_in_timeframe(self, user_id: int, days: int) -> int:
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT COUNT(*) FROM tickets
                    WHERE user_id = %s AND timestamp >= NOW() - INTERVAL '%s days';
                """, (user_id, days))
                result = await cur.fetchone()
                return result[0]

    async def rank_users_monthly(self) -> List[Tuple[int, int, int]]:
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT user_id,
                           COUNT(*) AS ticket_count,
                           RANK() OVER (ORDER BY COUNT(*) DESC) AS rank
                    FROM tickets
                    WHERE DATE_TRUNC('month', timestamp) = DATE_TRUNC('month', CURRENT_DATE)
                    GROUP BY user_id
                    ORDER BY ticket_count DESC;
                """)
                return await cur.fetchall()

    async def create_tables(self) -> None:
        """Creates necessary database tables if they don't exist"""
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS tickets (
                        id SERIAL PRIMARY KEY,
                        user_id BIGINT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON tickets(timestamp);")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON tickets(user_id);")


class RobloxAPI:
    """Handles all Roblox API interactions"""

    def __init__(self):
        self.headers = HEADERS

    async def get_user_info(self, discord_id: int) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                    f"https://api.blox.link/v4/public/guilds/{SERVER_ID}/discord-to-roblox/{discord_id}",
                    headers=self.headers
            ) as res:
                return await res.json()

    async def get_gamepass_ownership(self, user_id: int, gamepass_id: int) -> bool:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                    f"https://inventory.roblox.com/v1/users/{user_id}/items/1/{gamepass_id}/is-owned"
            ) as res:
                data = await res.json()
                if not isinstance(data, bool):
                    if "errors" in data:
                        return False
                return data

    async def get_past_usernames(self, roblox_id: int) -> List[str]:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                    f"https://users.roblox.com/v1/users/{roblox_id}/username-history"
            ) as r:
                response = await r.json()
                if "errors" in response:
                    return []
                return [entry["name"] for entry in response["data"]]

    async def get_avatar_url(self, avatar_url: str) -> Optional[str]:
        async with aiohttp.ClientSession() as session:
            async with session.get(avatar_url) as res:
                data = await res.json()
                try:
                    return data["data"][0]["imageUrl"]
                except (KeyError, IndexError):
                    return None


class ThreadManager:
    """Manages thread-related operations"""

    def __init__(self, db):
        self.db = db

    async def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        return await self.db.find_one({"thread_id": thread_id})

    async def claim_thread(self, thread_id: str, claimer_id: str, original_name: str) -> None:
        await self.db.insert_one({
            "thread_id": thread_id,
            "claimer": claimer_id,
            "original_name": original_name
        })

    async def unclaim_thread(self, thread_id: str) -> None:
        await self.db.find_one_and_delete({"thread_id": thread_id})

    async def transfer_thread(self, thread_id: str, new_claimer_id: str) -> None:
        await self.db.find_one_and_update(
            {"thread_id": thread_id},
            {"$set": {"claimer": new_claimer_id}}
        )


class DropDownChannels(discord.ui.Select):
    def __init__(self):
        options = [discord.SelectOption(label=name) for name in channel_options.keys()]
        super().__init__(
            placeholder="Select a channel...",
            options=options,
            min_values=1,
            max_values=1
        )

    async def callback(self, interaction: discord.Interaction):
        category_id = channel_options[self.values[0]]
        category = interaction.guild.get_channel(int(category_id))
        await interaction.channel.edit(category=category, sync_permissions=True)
        await interaction.response.edit_message(
            content="Moved channel successfully, thank the guides",
            view=None
        )


class DropDownView(discord.ui.View):
    def __init__(self):
        super().__init__()
        self.add_item(DropDownChannels())


def EmbedMaker(ctx: commands.Context, **kwargs) -> discord.Embed:
    """Creates consistent embeds for the bot"""
    color = colours.get(kwargs.pop("colour", "").lower(), 0x8E00FF)
    embed = discord.Embed(**kwargs, colour=color)
    embed.set_footer(
        text="City Airways",
        icon_url="https://cdn.discordapp.com/icons/788228600079843338/21fb48653b571db2d1801e29c6b2eb1d.png?size=4096"
    )
    return embed


def convert_to_seconds(text: str) -> int:
    """Converts time string to seconds"""
    return int(
        timedelta(
            **{
                UNITS.get(m.group("unit").lower(), "seconds"): float(m.group("val"))
                for m in re.finditer(
                    r"(?P<val>\d+(\.\d+)?)(?P<unit>[smhdw]?)",
                    text.replace(" ", ""),
                    flags=re.I,
                )
            }
        ).total_seconds()
    )


def unix_converter(seconds: int) -> int:
    """Converts seconds to unix timestamp"""
    return int((datetime.now() + timedelta(seconds=seconds)).timestamp())


def is_bypass():
    """Check if user has bypass permissions"""

    async def predicate(ctx):
        return ctx.author.id in BYPASS_LIST

    return commands.check(predicate)


async def check(ctx: commands.Context) -> bool:
    """Check if user can interact with thread"""
    if ctx.author.id in BYPASS_LIST:
        return True

    coll = ctx.bot.plugin_db.get_partition(ctx.bot.get_cog("GuidesCommittee"))
    thread = await coll.find_one({"thread_id": str(ctx.thread.channel.id)})

    if thread is not None:
        can_interact = ctx.author.bot or str(ctx.author.id) == thread["claimer"]
        if not can_interact and "⛔" not in [r.emoji for r in ctx.message.reactions]:
            await ctx.message.add_reaction("⛔")
        return can_interact

    if "⛔" not in [r.emoji for r in ctx.message.reactions]:
        await ctx.message.add_reaction("⛔")
    return False


async def create_database() -> aiopg.Pool:
    """Creates the database pool and ensures tables exist"""
    pool = await aiopg.create_pool(dsn)
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await cur.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON tickets(timestamp);")
            await cur.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON tickets(user_id);")
    return pool


class GuidesCommittee(commands.Cog):
    """Main cog for the Guides Committee functionality"""

    def __init__(self, bot):
        self.bot = bot
        self.db = self.bot.api.get_plugin_partition(self)
        self._frozen: Set[int] = set()
        self.logger = logging.getLogger('guides_committee')
        self.roblox_api = RobloxAPI()

    async def cog_load(self):
        """Initialize necessary components on cog load"""
        self.bot.pool = await create_database()
        self.db_manager = DatabaseManager(self.bot.pool)
        self.cooldown_mapping = AsyncCooldownMapping(self.bot.pool)
        self.thread_manager = ThreadManager(self.db)

        # Add checks to commands
        for cmd_name in ['reply', 'areply', 'fareply', 'freply', 'close']:
            cmd = self.bot.get_command(cmd_name)
            if cmd:
                cmd.add_check(check)

    async def cog_unload(self):
        """Cleanup when cog is unloaded"""
        # Remove checks
        for cmd_name in ['reply', 'areply', 'fareply', 'freply', 'close']:
            cmd = self.bot.get_command(cmd_name)
            if cmd and check in cmd.checks:
                cmd.remove_check(check)

        # Close pool
        await self.bot.pool.close()

    @commands.Cog.listener()
    async def on_thread_close(self, thread, closer, silent, delete_channel, message, scheduled):
        """Handle thread closure events"""
        if thread.recipient.id == closer.id:
            return await self._handle_self_close(closer)

        await self.db_manager.add_ticket(closer.id)
        await self._send_close_statistics(closer)

    async def _handle_self_close(self, closer: discord.Member) -> None:
        """Handle when a user closes their own ticket"""
        try:
            channel = closer.dm_channel or await closer.create_dm()
            await channel.send(
                "You closed your own ticket, it will not count towards your ticket count. "
                "A copy of this message is sent to management."
            )
        except discord.Forbidden:
            self.logger.warning(f"Could not send DM to user {closer.id}")

    async def _send_close_statistics(self, closer: discord.Member) -> None:
        """Send ticket statistics after closure"""
        try:
            channel = closer.dm_channel or await closer.create_dm()

            day_count = await self.db_manager.count_user_tickets(closer.id, 'day')
            week_count = await self.db_manager.count_user_tickets(closer.id, 'week')
            month_count = await self.db_manager.count_user_tickets(closer.id, 'month')

            cooldown = await self.cooldown_mapping.get_tickets_and_cooldown(closer.id)
            cooldown_time = cooldown[1]

            await channel.send(
                f"Congratulations on closing your ticket {closer}. This is your ticket number `{day_count}` today, "
                f"your ticket number `{week_count}` this week and your ticket number `{month_count}` this month. "
                f"Your cooldown is: `{cooldown_time:.1f}` seconds"
            )

            if day_count >= 8:
                await self._handle_high_ticket_count(closer, day_count, channel)

            if closer.id == 1208702357425102880:  # Special message for Ben
                await channel.send(
                    "Hi Ben, this is a special message I have in store for when you close a ticket. I just want to "
                    "extend my heartfelt congratulations, because this job you do is impressive."
                )

            if random.random() <= 0.3:  # 30% chance for motivational quote
                quote = random.choice(MOTIVATIONAL_QUOTES)
                embed = discord.Embed(color=colours["light_blue"], description=quote, title="Motivational Quote")
                await channel.send(embed=embed)

        except discord.Forbidden:
            self.logger.warning(f"Could not send DM to user {closer.id}")

    async def _handle_high_ticket_count(self, closer: discord.Member, day_count: int,
                                        channel: discord.DMChannel) -> None:
        """Handle high ticket count warnings"""
        if day_count == 8:
            await channel.send(
                "⚠**TICKET 8 WARNING**⚠\nClosing your 9th ticket will send a message to management where "
                "warnings/strikes/demotions might take place, if you have tickets currently claimed **UNCLAIM THEM**"
            )
        elif day_count > 8:
            warning_channel = await self.bot.fetch_channel(1311111724379672608)
            await warning_channel.send(
                f"⚠**MORE THAN 8 WARNING**⚠\n<@{closer.id}> closed more than 8 tickets in a day. "
                f"This is their ticket number `{day_count}` today"
            )

    @commands.dynamic_cooldown(lambda ctx: ctx.cog.cooldown_mapping.get_cooldown(ctx), commands.BucketType.user)
    @core.checks.thread_only()
    @core.checks.has_permissions(core.models.PermissionLevel.SUPPORTER)
    @commands.command()
    async def claim(self, ctx: commands.Context, bypass: str = None):
        """Claim a thread"""
        # Check claim timing
        created_time = ctx.channel.created_at.replace(tzinfo=timezone.utc)
        time_diff = (datetime.now(timezone.utc) - created_time).total_seconds() * 1000

        if time_diff < random.randint(1000, 1500):
            return await ctx.channel.send("Too fast, please try again")

        # Check daily limit
        day_count = await self.db_manager.count_user_tickets(ctx.author.id, 'day')
        if day_count == 8 and bypass != "bypass":
            embed = EmbedMaker(
                ctx,
                title="You have done 8 tickets today",
                description="You've done 8 tickets today! Doing more will cause management to be notified. "
                            "However if you wish to claim it run `.claim bypass`",
                colour="red"
            )
            return await ctx.send(embed=embed)

        # Check if thread is already claimed
        thread = await self.thread_manager.get_thread(str(ctx.thread.channel.id))
        if thread is not None:
            claimer = thread["claimer"]
            embed = EmbedMaker(
                ctx,
                title="Already Claimed",
                description=f"Already claimed by {(f'<@{claimer}>') if claimer != str(ctx.author.id) else 'you, dumbass'}",
                colour="red"
            )
            return await ctx.send(embed=embed)

        # Claim the thread
        await self.thread_manager.claim_thread(
            str(ctx.thread.channel.id),
            str(ctx.author.id),
            ctx.channel.name
        )

        try:
            await ctx.channel.edit(name=f"claimed-{ctx.author.display_name}")
            embed = EmbedMaker(
                ctx,
                title="Claimed",
                description=f"Claimed by {ctx.author.mention}",
                colour="green"
            )
            await ctx.send(embed=embed)
        except discord.Forbidden:
            await ctx.reply("I don't have permission to do that")

    @core.checks.thread_only()
    @core.checks.has_permissions(core.models.PermissionLevel.SUPPORTER)
    @commands.command()
    async def unclaim(self, ctx: commands.Context):
        """Unclaim a thread"""
        thread = await self.thread_manager.get_thread(str(ctx.thread.channel.id))

        if thread is None:
            embed = EmbedMaker(
                ctx,
                title="Already Unclaimed",
                description="This thread is not claimed, you can claim it!"
            )
            return await ctx.reply(embed=embed)

        if thread["claimer"] != str(ctx.author.id):
            embed = EmbedMaker(
                ctx,
                title="Unclaim Denied",
                description="You're not the claimer of this thread, don't anger chairwoman abbi"
            )
            return await ctx.reply(embed=embed)

        await self.thread_manager.unclaim_thread(str(ctx.thread.channel.id))

        try:
            await ctx.channel.edit(name=thread["original_name"])
            embed = EmbedMaker(
                ctx,
                title="Unclaimed",
                description=f"Unclaimed by {ctx.author.mention}",
                colour="green"
            )
            await ctx.reply(embed=embed)
        except discord.Forbidden:
            await ctx.reply("I don't have permission to do that")

    @core.checks.thread_only()
    @core.checks.has_permissions(core.models.PermissionLevel.SUPPORTER)
    @commands.command()
    async def takeover(self, ctx: commands.Context):
        """Take over a thread from another user"""
        if ctx.channel.id in self._frozen:
            return await ctx.send("Channel is frozen")

        thread = await self.thread_manager.get_thread(str(ctx.thread.channel.id))
        if thread["claimer"] == str(ctx.author.id):
            embed = EmbedMaker(
                ctx,
                title="Takeover Denied",
                description="You have literally claimed this yourself tf u doing",
                colour="red"
            )
            return await ctx.send(embed=embed)

        # Get role hierarchies
        taker_roles = [str(r.id) for r in ctx.author.roles if str(r.id) in ROLE_HIERARCHY]
        try:
            claimer = await ctx.guild.fetch_member(int(thread["claimer"]))
            claimed_roles = [str(r.id) for r in claimer.roles if str(r.id) in ROLE_HIERARCHY]
        except discord.NotFound:
            claimed_roles = []

        # Check if can takeover
        can_takeover = (
                not claimed_roles or
                ROLE_HIERARCHY.index(taker_roles[-1]) < ROLE_HIERARCHY.index(claimed_roles[-1]) or
                ctx.author.id in BYPASS_LIST
        )

        if can_takeover:
            await self.thread_manager.transfer_thread(str(ctx.thread.channel.id), str(ctx.author.id))
            try:
                await ctx.channel.edit(name=f"claimed-{ctx.author.display_name}")
                embed = EmbedMaker(
                    ctx,
                    title="Taken over successfully",
                    description=f"Takeover by {ctx.author.mention} successful"
                )
                await ctx.send(embed=embed)
            except discord.Forbidden:
                await ctx.reply("I couldn't change the channel name sorry")
        else:
            embed = EmbedMaker(
                ctx,
                title="Takeover Denied",
                description="Takeover denied since the claimer is your superior or the same rank as you"
            )
            await ctx.reply(embed=embed)

    @commands.command()
    @core.checks.has_permissions(core.models.PermissionLevel.MODERATOR)
    async def export(self, ctx: commands.Context):
        """Export monthly ticket rankings to CSV"""
        await ctx.message.add_reaction("<a:loading_f:1249799401958936576>")

        # Get ranking data
        results = await self.db_manager.rank_users_monthly()
        results = [list(i) for i in results]

        time = unix_converter(2.546 * len(results))
        msg = await ctx.reply(f"Started generation, estimated completion: <t:{time}:R>")

        # Convert Discord IDs to Roblox usernames
        rm_indices = []
        for idx, entry in enumerate(results):
            try:
                roblox_data = await self.roblox_api.get_user_info(entry[0])
                if "error" in roblox_data or "resolved" not in roblox_data:
                    raise ValueError("Invalid response")
                results[idx][0] = roblox_data["resolved"]["roblox"]["name"]
            except Exception as e:
                self.logger.error(f"Error processing user {entry[0]}: {e}")
                rm_indices.append(idx)
                await ctx.send(
                    f"Discord ID: {entry[0]} error, <@{entry[0]}> will not be included in pay, "
                    f"but if you need their ticket amount it is: `{entry[1]}`"
                )

        # Remove failed entries
        results = [item for idx, item in enumerate(results) if idx not in rm_indices]

        # Write CSV
        filename = f"monthly_ranking_{uuid.uuid4()}.csv"
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["ROBLOX Username", "Ticket Count", "Rank"])
            writer.writerows(results)

        await msg.delete()
        with open(filename, "rb") as f:
            await ctx.send(file=discord.File(f, filename=filename))

    @commands.command()
    @core.checks.thread_only()
    async def getinfo(self, ctx: commands.Context, member: discord.Member = None):
        """Get information about a user"""
        await ctx.message.add_reaction("<a:loading_f:1249799401958936576>")

        user_id = member.id if member else ctx.thread.recipient.id

        try:
            # Get Roblox data
            roblox_data = await self.roblox_api.get_user_info(user_id)
            if "error" in roblox_data:
                raise ValueError("Failed to get Roblox data")

            roblox_id = roblox_data["robloxID"]

            # Get gamepass ownership
            gamepass_status = {}
            for pass_name, pass_id in gamepasses.items():
                owns = await self.roblox_api.get_gamepass_ownership(roblox_id, pass_id)
                gamepass_status[pass_name] = owns

            # Get past usernames
            past_usernames = await self.roblox_api.get_past_usernames(roblox_id)

            # Create embed
            embed = discord.Embed(
                title=roblox_data["resolved"]["roblox"]["name"],
                url=roblox_data["resolved"]["roblox"]["profileLink"],
                colour=0x8E00FF,
                timestamp=datetime.now()
            )

            # Add fields
            embed.add_field(
                name="Discord",
                value=f"**ID**: {user_id}\n**Username**: {ctx.thread.recipient.name}\n"
                      f"**Display Name**: {ctx.thread.recipient.display_name}",
                inline=False
            )

            embed.add_field(
                name="ROBLOX",
                value=f"**ID**: {roblox_id}\n"
                      f"**Username**: {roblox_data['resolved']['roblox']['name']}\n"
                      f"**Display Name**: {roblox_data['resolved']['roblox']['displayName']}\n"
                      f"**Rank In Group**: {roblox_data['resolved']['roblox']['groupsv2']['8619634']['role']['name']} "
                      f"({roblox_data['resolved']['roblox']['groupsv2']['8619634']['role']['rank']})",
                inline=False
            )

            if past_usernames:
                embed.add_field(
                    name="ROBLOX PAST USERNAMES",
                    value="\n".join(past_usernames),
                    inline=False
                )

            gamepass_text = "\n".join(f"**{name}**: {EMOJI_VALUES[status]}"
                                      for name, status in gamepass_status.items())
            embed.add_field(name="Gamepasses", value=gamepass_text, inline=False)

            # Set thumbnail
            avatar_url = roblox_data["resolved"]["roblox"]["avatar"]["bustThumbnail"]
            async with aiohttp.ClientSession() as session:
                async with session.get(avatar_url) as res:
                    avatar_data = await res.json()
                    embed.set_thumbnail(url=avatar_data["data"][0]["imageUrl"])

            embed.set_footer(
                text=FOOTER,
                icon_url="https://cdn.discordapp.com/attachments/1208495821868245012/1249743898075463863/Logo.png"
            )

            await ctx.message.clear_reactions()
            await ctx.reply(embed=embed)

        except Exception as e:
            await ctx.message.clear_reactions()
            await ctx.reply(f"Error getting user info: {str(e)}")

    @commands.command()
    @core.checks.thread_only()
    @core.checks.has_permissions(core.models.PermissionLevel.SUPPORTER)
    async def mover(self, ctx: commands.Context):
        """Move the thread to a different channel"""
        view = DropDownView()
        await ctx.send("Choose a channel to move this ticket to", view=view)

    @commands.command()
    @core.checks.thread_only()
    @core.checks.has_permissions(core.models.PermissionLevel.SUPPORTER)
    async def remindme(self, ctx: commands.Context, time: str, *, message: str):
        """Set a reminder"""
        embed = EmbedMaker(
            ctx,
            title="Remind Me",
            description=f"I will remind you about {message} in {time}"
        )
        m = await ctx.reply(embed=embed)

        await asyncio.sleep(convert_to_seconds(time))
        await ctx.channel.send(f"{ctx.author.mention}, {message}")

        try:
            await m.delete()
            await ctx.author.send(message)
        except discord.Forbidden:
            pass

    @commands.command()
    @is_bypass()
    async def freeze(self, ctx: commands.Context):
        """Freeze a channel"""
        if ctx.channel.id not in self._frozen:
            self._frozen.add(ctx.channel.id)
        await ctx.send("This channel is frozen now, `takeover` is **disabled**, `transfer` is **enabled**.")

    @commands.command()
    @core.checks.thread_only()
    @is_bypass()
    async def transfer(self, ctx: commands.Context, user: discord.Member):
        """Transfer a thread to another user"""
        thread = await self.thread_manager.get_thread(str(ctx.thread.channel.id))

        if thread["claimer"] == str(user.id):
            embed = EmbedMaker(
                ctx,
                title="Transfer Denied",
                description=f"{user.mention} is already the thread claimer"
            )
            return await ctx.send(embed=embed)

        await self.thread_manager.transfer_thread(str(ctx.thread.channel.id), str(user.id))
        try:
            await ctx.channel.edit(name=f"claimed-{user.display_name}")
            embed = EmbedMaker(
                ctx,
                title="Transfer Successful",
                description=f"Thread transferred to {user.mention}"
            )
            await ctx.send(embed=embed)
        except discord.Forbidden:
            await ctx.reply("I couldn't change the channel name")

    async def cog_command_error(self, ctx: commands.Context, error: Exception):
        """Handle command errors"""
        if isinstance(error, commands.CommandOnCooldown):
            embed = EmbedMaker(
                ctx,
                title="On Cooldown",
                description=f"You can use this command again <t:{unix_converter(error.retry_after)}:R>",
                colour="red"
            )
            await ctx.send(embed=embed)
            return

        if isinstance(error, (commands.BadArgument, commands.BadUnionArgument)):
            await ctx.typing()
            await ctx.send(embed=discord.Embed(color=ctx.bot.error_color, description=str(error)))

        elif isinstance(error, commands.CommandNotFound):
            print("CommandNotFound: %s", error)

        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.send_help(ctx.command)

        elif isinstance(error, commands.CheckFailure):
            for check in ctx.command.checks:
                if not await check(ctx):
                    if hasattr(check, "fail_msg"):
                        await ctx.send(embed=discord.Embed(
                            color=ctx.bot.error_color,
                            description=check.fail_msg
                        ))
                    if hasattr(check, "permission_level"):
                        corrected_permission_level = ctx.bot.command_perm(ctx.command.qualified_name)
                        print(
                            "User %s does not have permission to use this command: `%s` (%s).",
                            ctx.author.name,
                            ctx.command.qualified_name,
                            corrected_permission_level.name
                        )
            print("CheckFailure: %s", error)

        elif isinstance(error, commands.DisabledCommand):
            print("DisabledCommand: %s is trying to run eval but it's disabled", ctx.author.name)

        elif isinstance(error, commands.CommandInvokeError):
            await ctx.send(embed=discord.Embed(
                color=ctx.bot.error_color,
                description=f"{str(error)}\nYou might be getting this error during **getinfo** if the user is either\n"
                            f"1. Not in the `main` server\n2. Has no linked account in bloxlink"
            ))
        else:
            await ctx.channel.send(f"{error}, {type(error)}")
            print("Unexpected exception:", error)

        try:
            await ctx.message.clear_reactions()
            await ctx.message.add_reaction("⛔")
        except Exception:
            pass

    @commands.command()
    @core.checks.has_permissions(core.models.PermissionLevel.SUPPORTER)
    async def owns(self, ctx: commands.Context, username: str, *, gamepass: str):
        """Check if a user owns a gamepass"""
        conversion_gamepass = False
        conversion_username = False

        try:
            gamepass = int(gamepass)
        except ValueError:
            gamepass = gamepass
            conversion_gamepass = True

        try:
            username_id = int(username)
        except ValueError:
            username = username
            conversion_username = True

        async with aiohttp.ClientSession() as session:
            if conversion_username:
                async with session.post(
                        "https://users.roblox.com/v1/usernames/users",
                        data=json.dumps({"usernames": [username], "excludeBannedUsers": True}),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if not bool(data["data"]):
                            embed = EmbedMaker(
                                ctx,
                                title="Wrong username",
                                description="Please try putting the right **ROBLOX** username"
                            )
                            return await ctx.reply(embed=embed)

                        if data["data"][0]["requestedUsername"] != username:
                            embed = EmbedMaker(
                                ctx,
                                title="Failed checks",
                                description="Error Checking, please try again"
                            )
                            return await ctx.reply(embed=embed)

                        username_id = data["data"][0]["id"]

            if conversion_gamepass:
                gamepass_id = find_most_similar(gamepass)

            async with session.get(
                    f"https://inventory.roblox.com/v1/users/{username_id}/items/1/{gamepass_id[1]}/is-owned"
            ) as resp:
                if resp.status == 200:
                    owns = await resp.json()

                    if not isinstance(owns, bool):
                        if "errors" in owns:
                            owns = False

                    if owns:
                        embed = EmbedMaker(
                            ctx,
                            title=f"{EMOJI_VALUES[True]} Ownership Verified",
                            description=f"{gamepass_id[0]} owned by {username}"
                        )
                    else:
                        embed = EmbedMaker(
                            ctx,
                            title=f"{EMOJI_VALUES[False]} Gamepass NOT Owned",
                            description=f"{gamepass_id[0]} **NOT** owned by {username}"
                        )

                    await ctx.reply(embed=embed)



async def setup(bot):
    await bot.add_cog(GuidesCommittee(bot))
