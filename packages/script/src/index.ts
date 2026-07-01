import semver from "semver"
import path from "path"

const rootPkgPath = path.resolve(import.meta.dir, "../../../package.json")
const rootPkg = await Bun.file(rootPkgPath).json()
const expectedBunVersion = rootPkg.packageManager?.split("@")[1]

if (!expectedBunVersion) {
  throw new Error("packageManager field not found in root package.json")
}

// relax version requirement
const expectedBunVersionRange = `^${expectedBunVersion}`

if (!semver.satisfies(process.versions.bun, expectedBunVersionRange)) {
  throw new Error(`This script requires bun@${expectedBunVersionRange}, but you are using bun@${process.versions.bun}`)
}

const opencodePkgPath = path.resolve(import.meta.dir, "../../../packages/opencode/package.json")
const opencodePkg = await Bun.file(opencodePkgPath).json()

export const Script = {
  get channel() {
    return "latest"
  },
  get version() {
    return opencodePkg.version
  },
  get release(): boolean {
    return !!process.env["OPENCODE_RELEASE"]
  },
}
console.log(`opencode script`, JSON.stringify(Script, null, 2))
