from nats import connect, errors
from nats.aio.client import Client as NATS
from nats.js import JetStreamContext
from nats.js.api import AckPolicy, ConsumerConfig, DeliverPolicy, StreamConfig


class JetStreamManager:
    def __init__(self, servers="nats://localhost:4222"):
        self.servers = servers
        self.nc: NATS | None = None
        self.js: JetStreamContext | None = None

    async def connect(self):
        """Establish NATS connection and JetStream context"""
        self.nc = await connect(self.servers)
        self.js = self.nc.jetstream()
        return self.js

    async def ensure_stream(self, stream_name: str, subjects: list, **kwargs):
        if self.js is None:
            raise RuntimeError("Jetstream not initialized")

        try:
            stream = await self.js.stream_info(stream_name)
            config = StreamConfig(
                name=stream_name,
                subjects=subjects,
                **{**stream.config.dict(), **kwargs}
            )
            return await self.js.update_stream(config)
        except:
            config = StreamConfig(name=stream_name, subjects=subjects, **kwargs)
            return await self.js.add_stream(config)

    async def create_consumer(
        self,
        subject: str,
        stream_name: str,
        consumer_name: str,
        description: str,
        ack_policy: AckPolicy = AckPolicy.EXPLICIT,
        deliver_policy: DeliverPolicy = DeliverPolicy.ALL,
    ):
        if self.js is None:
            raise RuntimeError("Jetstream not intialized")

        config = ConsumerConfig(
            durable_name=consumer_name,
            ack_policy=ack_policy,
            deliver_policy=deliver_policy,
            description=description,
        )
        return await self.js.pull_subscribe(
            stream=stream_name, subject=subject, config=config, durable=consumer_name
        )

    async def shutdown(self):
        """Clean connection closure"""
        if self.nc is None:
            raise RuntimeError("NATS client not initialized")

        await self.nc.drain()

    async def process_messages(
        self, subscriber: JetStreamContext.PullSubscription, batch_size=10, timeout=30
    ):
        while True:
            try:
                messages = await subscriber.fetch(batch_size, timeout=timeout)
                for msg in messages:
                    yield msg
            except errors.TimeoutError:
                break
