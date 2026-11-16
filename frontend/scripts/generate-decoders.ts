/** Compile the Rust-produced OpenAPI schemas into reusable Effect decoders. */
import { readFileSync, writeFileSync } from "node:fs";
import { transformSync } from "@babel/core";
import * as JsonSchema from "effect/JsonSchema";
import * as Representation from "effect/SchemaRepresentation";
import * as Schema from "effect/Schema";

const document = Schema.decodeUnknownSync(
  Schema.Struct({
    openapi: Schema.String.check(Schema.isStartingWith("3.1.")),
    components: Schema.Struct({
      schemas: Schema.Record(
        Schema.String,
        Schema.Record(Schema.String, Schema.Unknown),
      ),
    }),
  }),
)(
  JSON.parse(readFileSync(new URL("../openapi.json", import.meta.url), "utf8")),
);

// All roots share one conversion and one reference graph. Schema import refuses
// unsupported constraints instead of substituting weaker validation.
const roots = [
  "JobInfo",
  "JobListItem",
  "HealthResponse",
  "FileStatusEntry",
  "StatusMessageResponse",
] as const;
const properties = Object.fromEntries(
  roots.map((name) => [name, { $ref: `#/components/schemas/${name}` }]),
);
const imported = Representation.fromJsonSchemaDocument(
  JsonSchema.fromSchemaOpenApi3_1({
    type: "object",
    properties,
    required: roots,
    $defs: document.components.schemas,
  }),
  {
    // Explanatory prose remains in the canonical OpenAPI document. The browser
    // carries validation constraints and reference identities only.
    onEnter: ({
      description: _description,
      title: _title,
      examples: _examples,
      ...schema
    }) => schema,
  },
);
const code = Representation.toCodeDocument(
  Representation.toMultiDocument(Representation.toRepresentation(imported.ast)),
);
const references = [
  ...code.references.nonRecursives.map((ref) => [ref.$ref, ref.code] as const),
  ...Object.entries(code.references.recursives),
];
const declarations = references.map(
  ([name, value]) =>
    `export type ${name} = ${value.Type};\nexport const ${name}: Schema.Codec<${name}> = ${value.runtime};`,
);
const artifacts = code.artifacts.map((artifact) => {
  switch (artifact._tag) {
    case "Import":
      return artifact.importDeclaration;
    case "Symbol":
      return `const ${artifact.identifier} = ${artifact.code.runtime};`;
    case "Enum":
      return artifact.code.runtime;
  }
});
const source = [
  "/** Generated from openapi.json. Regenerate through npm run generate:decoders. */",
  'import * as Schema from "effect/Schema";',
  ...artifacts,
  ...declarations,
].join("\n\n");
// The existing Babel toolchain prints readable TypeScript without stripping it.
const formatted = Schema.decodeUnknownSync(Schema.Struct({ code: Schema.String }))(
  transformSync(source, {
    filename: "decoders.ts",
    configFile: false,
    babelrc: false,
    parserOpts: { plugins: ["typescript"] },
    generatorOpts: { compact: false },
  }),
);
writeFileSync(new URL("../src/generated/decoders.ts", import.meta.url), formatted.code + "\n");
