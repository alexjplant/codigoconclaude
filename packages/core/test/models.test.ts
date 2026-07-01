import { describe, expect, beforeAll, afterAll } from "bun:test"
import { Effect, Layer } from "effect"
import { AppNodeBuilder } from "@opencode-ai/core/effect/app-node-builder"
import { Flag } from "@opencode-ai/core/flag/flag"
import { Global } from "@opencode-ai/core/global"
import { ModelsDev } from "@opencode-ai/core/models-dev"
import { it } from "./lib/effect"
import { rm, writeFile, mkdir } from "fs/promises"
import path from "path"

const ORIGINAL_MODELS_PATH = Flag.OPENCODE_MODELS_PATH
const cacheFile = path.join(Global.Path.cache, "models.json")

beforeAll(async () => {
  Flag.OPENCODE_MODELS_PATH = undefined
  await rm(cacheFile, { force: true })
})
afterAll(async () => {
  Flag.OPENCODE_MODELS_PATH = ORIGINAL_MODELS_PATH
  await rm(cacheFile, { force: true })
})

const fixture: Record<string, ModelsDev.Provider> = {
  acme: {
    id: "acme",
    name: "Acme",
    env: ["ACME_API_KEY"],
    models: {
      "acme-1": {
        id: "acme-1",
        name: "Acme One",
        release_date: "2026-01-01",
        attachment: false,
        reasoning: false,
        temperature: true,
        tool_call: true,
        limit: { context: 128000, output: 8192 },
      },
    },
  },
}

const writeCache = (data: object) =>
  Effect.promise(async () => {
    await mkdir(Global.Path.cache, { recursive: true })
    await writeFile(cacheFile, JSON.stringify(data))
  })

const buildLayer = () =>
  Layer.fresh(
    AppNodeBuilder.build(ModelsDev.node),
  )

const provided = <A, E>(eff: Effect.Effect<A, E, ModelsDev.Service>) =>
  eff.pipe(Effect.provide(buildLayer()))

describe("ModelsDev Service", () => {
  it.live("get() returns providers from disk when cache file exists", () =>
    Effect.gen(function* () {
      yield* writeCache(fixture)
      const result = yield* provided(ModelsDev.Service.use((s) => s.get()))
      expect(result).toEqual(fixture)
    }),
  )

  it.live("get() returns empty catalog when disk is empty and no bundled snapshot", () =>
    Effect.gen(function* () {
      yield* Effect.promise(() => rm(cacheFile, { force: true }))
      const result = yield* provided(ModelsDev.Service.use((s) => s.get()))
      expect(result).toEqual({})
    }),
  )

  it.live("get() caches across calls (later disk writes are ignored until process restart)", () =>
    provided(
      Effect.gen(function* () {
        yield* writeCache(fixture)
        const svc = yield* ModelsDev.Service
        const a = yield* svc.get()
        yield* writeCache({})
        const b = yield* svc.get()
        expect(a).toEqual(fixture)
        expect(b).toEqual(fixture)
      }),
    ),
  )
})
