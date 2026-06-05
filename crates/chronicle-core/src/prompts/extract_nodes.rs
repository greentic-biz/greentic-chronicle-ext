// Ported from graphiti_core/prompts/extract_nodes.py @ 34f56e65 (v0.29.1)
//
// Fidelity notes:
// - Every system prompt has DO_NOT_ESCAPE_UNICODE appended, replicating upstream's
//   VersionWrapper which appends it to the SYSTEM message of every prompt version.
// - Upstream f-string user prompts begin with a literal '\n' (the newline right after
//   the opening triple-quote) and we preserve it.
// - `context['entity_types']` is interpolated RAW upstream (no to_prompt_json). At the
//   call site it is a Python `list[dict]`, so `{context['entity_types']}` renders Python's
//   `str(list)` form (single-quoted keys). True byte-identical reproduction of that VALUE
//   is impossible in Rust; we render it via to_prompt_json (JSON, double-quoted). The
//   surrounding prompt text is byte-identical. This is the one documented value-rendering
//   divergence; see the project fidelity ledger.
// - `previous_episodes` is rendered with to_prompt_json (upstream calls
//   to_prompt_json([ep for ep in context['previous_episodes']])).
// - `custom_extraction_instructions` is the already-resolved string the caller passes
//   (upstream computes `custom_extraction_instructions or ''` at the call site), so the
//   context field carries the final text (empty string when absent).

use crate::llm::Message;
use crate::prompts::helpers::{DO_NOT_ESCAPE_UNICODE, to_prompt_json};

/// Context for [`extract_message`].
pub struct ExtractMessageContext<'a> {
    /// Upstream `context['entity_types']` (a `list[dict]` rendered raw upstream).
    pub entity_types: &'a serde_json::Value,
    /// Upstream `context['previous_episodes']`, rendered via to_prompt_json.
    pub previous_episodes: &'a [String],
    /// Upstream `context['episode_content']`.
    pub episode_content: &'a str,
    /// Upstream `context['custom_extraction_instructions']` (already resolved to ''
    /// when the caller has none).
    pub custom_extraction_instructions: &'a str,
}

/// Verbatim port of upstream `extract_message`.
pub fn extract_message(ctx: &ExtractMessageContext<'_>) -> Vec<Message> {
    let sys_prompt = format!(
        "You are an entity extraction specialist for conversational messages. \
NEVER extract abstract concepts, feelings, or generic words.{DO_NOT_ESCAPE_UNICODE}"
    );

    let entity_types = to_prompt_json(ctx.entity_types);
    let previous_episodes = to_prompt_json(&ctx.previous_episodes);
    let episode_content = ctx.episode_content;
    let custom_extraction_instructions = ctx.custom_extraction_instructions;

    let user_prompt = format!(
        r#"
NEVER extract any of the following:
- Pronouns (you, me, I, he, she, they, we, us, it, them, him, her, this, that, those)
- Abstract concepts or feelings (joy, balance, growth, resilience, happiness, passion, motivation)
- Generic common nouns or bare object words (day, life, people, work, stuff, things, food, time,
  way, tickets, supplies, clothes, keys, gear)
- Generic media/content nouns unless uniquely identified in the node name itself (photo, pic, picture,
  image, video, post, story)
- Generic event/activity nouns unless uniquely identified in the node name itself (event, game, meeting,
  class, workshop, competition)
- Broad institutional nouns unless explicitly named or uniquely qualified (government, school, company,
  team, office)
- Ambiguous bare nouns whose meaning depends on sentence context rather than the node name itself
- Sentence fragments or clauses ("what you really care about", "results of that effort")
- Adjectives or descriptive phrases ("amazing", "something different", "new hair color")
- Duplicate references to the same real-world entity. Extract each entity at most once per message,
  even if it appears multiple times or both as a speaker label and in the body text.
- Bare relational or kinship terms (dad, mom, mother, father, sister, brother, husband, wife,
  spouse, son, daughter, uncle, aunt, cousin, grandma, grandpa, friend, boss, teacher, neighbor,
  roommate) and bare animal/pet words (dog, cat, pet, puppy, kitten). These are too generic on
  their own. Instead, qualify them with the possessor: extract "Nisha's dad" not "dad",
  "Jordan's dog" not "dog".
- Bare generic objects that cannot be meaningfully qualified with a possessor, brand, or
  distinguishing detail (e.g., NEVER extract "supplies" from "I picked up some supplies")

Your task is to extract **entity nodes** that are **explicitly** mentioned in the CURRENT MESSAGE.
Pronoun references such as he/she/they or this/that/those should be disambiguated to the names of the
reference entities. Only extract distinct entities from the CURRENT MESSAGE.

<ENTITY TYPES>
{entity_types}
</ENTITY TYPES>

<PREVIOUS MESSAGES>
{previous_episodes}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{episode_content}
</CURRENT MESSAGE>

1. **Speaker Extraction**: Always extract the speaker (the part before the colon `:` in each dialogue line) as the first entity node.
   - If the speaker is mentioned again in the message, treat both mentions as a **single entity**.

2. **Entity Identification**:
   - Extract named entities and specific, concrete things that are **explicitly** mentioned in the CURRENT MESSAGE.
   - Only extract entities that are specific enough to be uniquely identifiable. Ask: "Could this have its own Wikipedia article or database entry, OR is it specific enough to distinguish from other items of the same category within this conversation?"
   - For objects, possessions, and physical items, extract when they are specific enough
     to distinguish from other items of the same category. SHOULD be extracted:
     - Brand-named items ("Gamecube", "Ford Mustang", "Moen faucet")
     - Qualified items ("wool coat", "red and purple lighting", "cracked windshield",
       "dog leash")
     - Items with a concrete distinguishing descriptor (color, material, size, model,
       owner, specific use)
     Should NOT be extracted:
     - Bare head nouns alone ("car", "coat", "game", "lighting", "windshield")
   - When a speaker or named person refers to a relative, pet, or associate using a bare term
     (e.g., "my dad", "his cat"), extract the entity qualified with the possessor's name
     (e.g., "Nisha's dad", "Jordan's cat"). Do NOT extract the bare term alone.
   - **Exclude** entities mentioned only in the PREVIOUS MESSAGES (they are for context only).

3. **Entity Classification**:
   - Use the descriptions in ENTITY TYPES to classify each extracted entity.
   - Assign the appropriate `entity_type_id` for each one.

4. **Exclusions**:
   - Do NOT extract entities representing relationships or actions.
   - Do NOT extract dates, times, or other temporal information — these will be handled separately.
   - When in doubt, do NOT extract.

5. **Specificity**:
   - Always use the **most specific form** mentioned in the message. If the message says "road cycling",
     extract "road cycling" not "cycling". If it says "wool coat", extract "wool coat" not "coat".
   - When context makes an object's type clear, include that context in the name. For example, if the
     message mentions forgetting a leash while discussing a dog walk, extract "dog leash" not "leash".
   - If a phrase would not be distinguishable when read alone later, do NOT extract it.

6. **Formatting**:
   - Be **explicit and unambiguous** in naming entities (e.g., use full names when available).

<EXAMPLE>
Message: "Jordan: We just moved to Denver last month. My spouse started a new role at Lockheed Martin and I enrolled in a ceramics workshop at the Belmont Arts Center."
Good extractions: "Jordan" (speaker), "Denver" (Location), "Lockheed Martin" (Organization), "Belmont Arts Center" (Location), "ceramics" (Topic)
Do NOT extract: "spouse" (generic reference — extract only if named), "new role" (not an entity), "last month" (temporal), "we" (pronoun)
</EXAMPLE>

<EXAMPLE>
Message: "Nisha: My dad is visiting next week. He loves walking his dogs in Riverside Park."
Good extractions: "Nisha" (speaker), "Nisha's dad" (Person), "Riverside Park" (Location)
Do NOT extract: "dad" (bare relational term — qualify as "Nisha's dad"), "dogs" (bare animal word — no specific identity), "next week" (temporal)
</EXAMPLE>

<EXAMPLE>
Message: "Mary: I forgot Trigger's leash so I couldn't take him on a dog walk. After that I went road cycling in my new wool coat."
Good extractions: "Mary" (speaker), "Trigger" (animal name), "dog leash" (Object), "road cycling" (Topic), "wool coat" (Object)
Do NOT extract: "leash" (too generic — use "dog leash"), "cycling" (too generic — use "road cycling"), "coat" (too generic — use "wool coat"), "dog walk" (activity, not an entity)
</EXAMPLE>

<EXAMPLE>
Message: "Nate: My gaming room has red and purple lighting and I mostly play on a Gamecube. Last week the windshield on my Mustang got cracked."
Good extractions: "Nate" (speaker), "gaming room" (Object), "red and purple lighting" (Object), "Gamecube" (Object), "Mustang" (Object), "cracked windshield" (Object)
Do NOT extract: "lighting" (bare head noun — use "red and purple lighting"), "windshield" (bare head noun — use "cracked windshield"), "week" (temporal)
</EXAMPLE>

<EXAMPLE>
Message: "Alex: I shared a pic from the game after the event."
Good extractions: "Alex" (speaker)
Do NOT extract: "pic" (generic media noun), "game" (generic event noun), "event" (generic event noun)
</EXAMPLE>

<EXAMPLE>
Message: "Jordan: We won by a tight score. Scoring that last basket felt incredible."
Good extractions: "Jordan" (speaker)
Do NOT extract: "basket" (ambiguous bare noun that depends on sentence context)
</EXAMPLE>

{custom_extraction_instructions}
"#
    );

    vec![Message::system(sys_prompt), Message::user(user_prompt)]
}

/// Context for [`extract_json`].
pub struct ExtractJsonContext<'a> {
    /// Upstream `context['entity_types']` (rendered raw upstream; see module note).
    pub entity_types: &'a serde_json::Value,
    /// Upstream `context['source_description']`.
    pub source_description: &'a str,
    /// Upstream `context['episode_content']`.
    pub episode_content: &'a str,
    /// Upstream `context['custom_extraction_instructions']` (resolved to '' when absent).
    pub custom_extraction_instructions: &'a str,
}

/// Verbatim port of upstream `extract_json`.
pub fn extract_json(ctx: &ExtractJsonContext<'_>) -> Vec<Message> {
    let sys_prompt = format!(
        "You are an entity extraction specialist for JSON data. \
NEVER extract abstract concepts, dates, or generic field values.{DO_NOT_ESCAPE_UNICODE}"
    );

    let entity_types = to_prompt_json(ctx.entity_types);
    let source_description = ctx.source_description;
    let episode_content = ctx.episode_content;
    let custom_extraction_instructions = ctx.custom_extraction_instructions;

    let user_prompt = format!(
        r#"
NEVER extract:
- Date, time, or timestamp values
- Abstract concepts or generic field values (e.g., "true", "active", "pending")
- Numeric IDs or codes that are not meaningful entity names
- Bare relational or kinship terms (e.g., "spouse", "parent", "pet") — only extract if qualified
  with a possessor name
- Bare generic objects or common nouns (e.g., "supplies", "tickets", "gear") — only extract if
  qualified with a distinguishing detail
- Generic media/content nouns unless uniquely identified in the value itself (photo, pic, picture,
  image, video, post, story)
- Generic event/activity nouns unless uniquely identified in the value itself (event, game, meeting,
  class, workshop, competition)
- Broad institutional nouns unless explicitly named or uniquely qualified (government, school, company,
  team, office)
- Ambiguous bare nouns whose meaning depends on surrounding text rather than the extracted value itself

Extract entities from the JSON and classify each using the ENTITY TYPES above.

<ENTITY TYPES>
{entity_types}
</ENTITY TYPES>

<SOURCE DESCRIPTION>
{source_description}
</SOURCE DESCRIPTION>

<JSON>
{episode_content}
</JSON>

Guidelines:
1. Extract the primary entity the JSON represents (e.g., a "name" or "user" field).
2. Extract named entities referenced in other properties throughout the JSON structure.
3. Only extract entities specific enough to be uniquely identifiable.
4. Be explicit in naming entities — use full names when available.
5. Use the most specific form present in the data (e.g., "road cycling" not "cycling").
6. If a value would not be meaningful and distinguishable when read alone later, do NOT extract it.

{custom_extraction_instructions}

<EXAMPLE>
JSON: {{"user": "Jordan Lee", "company": "Acme Corp", "role": "engineer", "start_date": "2024-01-15", "location": "Denver", "active": true}}
Good extractions: "Jordan Lee" (Person), "Acme Corp" (Organization), "Denver" (Location)
Do NOT extract: "engineer" (role, not an entity), "2024-01-15" (date), "true" (field value)
</EXAMPLE>

<EXAMPLE>
JSON: {{"author": "Alex", "attachment_type": "photo", "event_name": "event", "agency": "government"}}
Good extractions: "Alex" (Person)
Do NOT extract: "photo" (generic media noun), "event" (generic event noun), "government" (broad institutional noun)
</EXAMPLE>
"#
    );

    vec![Message::system(sys_prompt), Message::user(user_prompt)]
}

/// Context for [`extract_text`].
pub struct ExtractTextContext<'a> {
    /// Upstream `context['entity_types']` (rendered raw upstream; see module note).
    pub entity_types: &'a serde_json::Value,
    /// Upstream `context['episode_content']`.
    pub episode_content: &'a str,
    /// Upstream `context['custom_extraction_instructions']` (resolved to '' when absent).
    pub custom_extraction_instructions: &'a str,
}

/// Verbatim port of upstream `extract_text`.
pub fn extract_text(ctx: &ExtractTextContext<'_>) -> Vec<Message> {
    let sys_prompt = format!(
        "You are an entity extraction specialist for unstructured text. \
NEVER extract abstract concepts, feelings, or generic words.{DO_NOT_ESCAPE_UNICODE}"
    );

    let entity_types = to_prompt_json(ctx.entity_types);
    let episode_content = ctx.episode_content;
    let custom_extraction_instructions = ctx.custom_extraction_instructions;

    let user_prompt = format!(
        r#"
NEVER extract:
- Pronouns (you, me, he, she, they, it, them, him, her, we, us, this, that, those)
- Abstract concepts (joy, balance, growth, resilience, passion, motivation)
- Generic common nouns or bare object words (day, life, people, work, stuff, things, food, time,
  tickets, supplies, clothes, keys, gear)
- Generic media/content nouns unless uniquely identified in the node name itself (photo, pic, picture,
  image, video, post, story)
- Generic event/activity nouns unless uniquely identified in the node name itself (event, game, meeting,
  class, workshop, competition)
- Broad institutional nouns unless explicitly named or uniquely qualified (government, school, company,
  team, office)
- Ambiguous bare nouns whose meaning depends on sentence context rather than the node name itself
- Sentence fragments or clauses as entity names
- Bare relational or kinship terms (dad, mom, sister, brother, spouse, friend, boss, pet, dog,
  cat) unless qualified with a possessor (e.g., "Nisha's dad" is acceptable, "dad" alone is not)
- Bare generic objects that cannot be meaningfully qualified with a possessor, brand, or
  distinguishing detail (e.g., NEVER extract "supplies" from "I picked up some supplies")

Extract entities from the TEXT that are **explicitly mentioned**.
For each entity, classify it using the ENTITY TYPES above.
Only extract entities specific enough to be uniquely identifiable — ask: "Could this have its own Wikipedia article or database entry?"

<ENTITY TYPES>
{entity_types}
</ENTITY TYPES>

<TEXT>
{episode_content}
</TEXT>

Guidelines:
1. Extract named entities and specific, concrete things.
2. Do not create nodes for relationships or actions.
3. Do not create nodes for temporal information like dates, times or years.
4. Be explicit in node names, using full names and avoiding abbreviations.
5. Always use the most specific form from the text (e.g., "road cycling" not "cycling",
   "wool coat" not "coat"). Include qualifying context when it's clear from the text.
6. When the text refers to a person's relative, pet, or associate by a bare term, qualify the
   entity with the possessor's name (e.g., "Dr. Osei's colleague" not "colleague").
7. If a phrase would not be meaningful and distinguishable when read alone later, do NOT extract it.
8. When in doubt, do NOT extract.

{custom_extraction_instructions}

<EXAMPLE>
Text: "Dr. Amara Osei presented her migraine study results at the AAN conference. The study tracked 340 patients using a new CGRP combination protocol."
Good extractions: "Dr. Amara Osei" (Person), "AAN" (Organization), "migraine study" (Topic), "CGRP combination protocol" (Object)
Do NOT extract: "results" (generic noun), "340" (number), "patients" (generic noun), "conference" (generic without a specific name)
</EXAMPLE>

<EXAMPLE>
Text: "Alex shared a pic after the event and said scoring the last basket felt incredible."
Good extractions: "Alex" (Person)
Do NOT extract: "pic" (generic media noun), "event" (generic event noun), "basket" (ambiguous bare noun)
</EXAMPLE>
"#
    );

    vec![Message::system(sys_prompt), Message::user(user_prompt)]
}

/// Context for [`extract_attributes`].
pub struct ExtractAttributesContext<'a> {
    /// Upstream `context['previous_episodes']`, rendered via to_prompt_json.
    pub previous_episodes: &'a [String],
    /// Upstream `context['episode_content']`, rendered via to_prompt_json.
    pub episode_content: &'a serde_json::Value,
    /// Upstream `context['node']` is a Python `dict` interpolated RAW (`{context['node']}`),
    /// so upstream emits Python's `str(dict)` form with single quotes (e.g. `{'name': 'Mary'}`).
    /// True byte-identical reproduction of Python's dict repr is impossible in Rust; we render
    /// node via to_prompt_json (well-formed JSON, double-quoted). This is a documented
    /// value-rendering divergence — the surrounding prompt text is byte-identical.
    pub node: &'a serde_json::Value,
}

/// Verbatim port of upstream `extract_attributes`.
pub fn extract_attributes(ctx: &ExtractAttributesContext<'_>) -> Vec<Message> {
    let sys_prompt = format!(
        "You are an entity attribute extraction specialist. \
You ONLY emit attribute values that are explicitly stated in MESSAGES or \
already present on the ENTITY. You output strictly the JSON specified by the \
response schema — no reasoning, no explanation, no commentary in any field.{DO_NOT_ESCAPE_UNICODE}"
    );

    let previous_episodes = to_prompt_json(&ctx.previous_episodes);
    let episode_content = to_prompt_json(ctx.episode_content);
    let node = to_prompt_json(ctx.node);

    // Upstream user content uses f"""\ (a leading backslash) so the string does NOT
    // start with a newline; the first character is 'G' of "Given".
    let user_prompt = format!(
        r#"Given the MESSAGES and the following ENTITY, update its attributes.

HARD RULES — violating any of these is a failure:

1. Each attribute value MUST be one of:
   (a) a clean value copied or directly normalized from text in MESSAGES,
   (b) the existing value already on the ENTITY (preserved unchanged), or
   (c) null / omitted, when neither (a) nor (b) applies.

2. NEVER write reasoning, justification, or commentary into any field. Specifically:
   - NEVER include parenthetical explanations like "(implied by ...)", "(Context: ...)",
     "(not explicitly stated ...)", "(based on ...)".
   - NEVER include first-person or deliberative phrases like "I should...", "However...",
     "Sticking to...", "Since no...", "the instruction is to...", "must be kept...",
     "if no value is present...".
   - NEVER list alternatives or candidates inside one field ("X, or Y, or maybe Z").
   - NEVER explain why a value is null. If unknown, set the field to null and stop.

3. Each attribute schema description (e.g. an "Industry sector" field whose description
   reads "Industry classification, single word where possible") tells you the FORMAT a
   real value should take. The description text is NEVER itself a value. NEVER copy
   schema description text into the field.

4. The literal strings "null", "N/A", "Not specified", "unknown", "none", "not provided",
   or any sentence describing absence are NOT valid values. If no value is supported by
   MESSAGES, set the field to null (or omit it) — do not write a sentence.

5. Each attribute value must be a short, well-formed instance of the type the field
   describes (a phone number, an industry name, a URL, a postal address). If you cannot
   produce a clean value of that type from MESSAGES, the field is null.

6. NEVER infer attribute values from the entity's name, from related entities, from
   generic world knowledge, or from prior summaries. Only verbatim or directly normalized
   text from MESSAGES qualifies as a new value.

7. If MESSAGES contain no information about an attribute, leave the existing entity
   value unchanged. If the entity has no existing value, the field is null.

EXAMPLES

ENTITY: {{"name": "Sam Rivera", "phones": "415-555-0142"}}
MESSAGES contain no phone information for Sam.
GOOD → "phones": "415-555-0142"   (preserved existing value)
BAD  → "phones": "415-555-0142 (implied by original entity, but no new information in
        messages, retaining original value as per instruction...)"

ENTITY: {{"name": "Northwind", "industry": null}}
MESSAGES mention Northwind only as the platform some content was posted to.
GOOD → "industry": null   (no explicit industry classification was stated)
BAD  → "industry": "Content platform, SaaS (implied by usage context, though not stated
        explicitly as industry classification...)"

ENTITY: {{"name": "Priya"}}
MESSAGES contain no phone for Priya, but discuss a project she contributed to.
GOOD → "phones": null
BAD  → "phones": "Worked with Lin and Marco on the Q3 launch..."   (off-topic content dump)

<MESSAGES>
{previous_episodes}
{episode_content}
</MESSAGES>

<ENTITY>
{node}
</ENTITY>
"#
    );

    vec![Message::system(sys_prompt), Message::user(user_prompt)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Role;
    use crate::prompts::helpers::DO_NOT_ESCAPE_UNICODE;

    fn entity_types() -> serde_json::Value {
        serde_json::json!([{"entity_type_id": 0, "entity_type_name": "Person"}])
    }

    #[test]
    fn extract_message_renders_verbatim() {
        let prev = vec!["Jordan: hi".to_string()];
        let ctx = ExtractMessageContext {
            entity_types: &entity_types(),
            previous_episodes: &prev,
            episode_content: "Mary: I love road cycling.",
            custom_extraction_instructions: "",
        };
        let msgs = extract_message(&ctx);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].role, Role::User);
        assert!(msgs[0].content.ends_with(DO_NOT_ESCAPE_UNICODE));
        assert!(
            msgs[0]
                .content
                .contains("You are an entity extraction specialist for conversational messages.")
        );
        let u = &msgs[1].content;
        assert!(u.contains(
            "Your task is to extract **entity nodes** that are **explicitly** mentioned in the CURRENT MESSAGE."
        ));
        assert!(u.contains(
            "1. **Speaker Extraction**: Always extract the speaker (the part before the colon `:` in each dialogue line) as the first entity node."
        ));
        assert!(u.contains("- **Exclude** entities mentioned only in the PREVIOUS MESSAGES (they are for context only)."));
        assert!(u.contains("Mary: I love road cycling."));
    }

    #[test]
    fn extract_text_renders_verbatim() {
        let prev: Vec<String> = vec![];
        let _ = &prev;
        let ctx = ExtractTextContext {
            entity_types: &entity_types(),
            episode_content: "Dr. Amara Osei presented at AAN.",
            custom_extraction_instructions: "",
        };
        let msgs = extract_text(&ctx);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].role, Role::User);
        assert!(msgs[0].content.ends_with(DO_NOT_ESCAPE_UNICODE));
        assert!(
            msgs[0]
                .content
                .contains("You are an entity extraction specialist for unstructured text.")
        );
        let u = &msgs[1].content;
        assert!(u.contains("Extract entities from the TEXT that are **explicitly mentioned**."));
        assert!(u.contains("2. Do not create nodes for relationships or actions."));
        assert!(u.contains(
            "Only extract entities specific enough to be uniquely identifiable — ask: \"Could this have its own Wikipedia article or database entry?\""
        ));
        assert!(u.contains("Dr. Amara Osei presented at AAN."));
    }

    #[test]
    fn extract_json_renders_verbatim() {
        let ctx = ExtractJsonContext {
            entity_types: &entity_types(),
            source_description: "a user record",
            episode_content: "{\"user\": \"Jordan Lee\"}",
            custom_extraction_instructions: "",
        };
        let msgs = extract_json(&ctx);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].role, Role::User);
        assert!(msgs[0].content.ends_with(DO_NOT_ESCAPE_UNICODE));
        assert!(
            msgs[0]
                .content
                .contains("You are an entity extraction specialist for JSON data.")
        );
        let u = &msgs[1].content;
        assert!(u.contains(
            "Extract entities from the JSON and classify each using the ENTITY TYPES above."
        ));
        assert!(u.contains("1. Extract the primary entity the JSON represents (e.g., a \"name\" or \"user\" field)."));
        assert!(u.contains("<SOURCE DESCRIPTION>\na user record\n</SOURCE DESCRIPTION>"));
        assert!(u.contains("{\"user\": \"Jordan Lee\"}"));
    }

    #[test]
    fn extract_attributes_renders_verbatim() {
        let prev = vec!["Mary: hello".to_string()];
        let episode = serde_json::json!("Mary works at Acme.");
        let node = serde_json::json!({"name": "Mary"});
        let ctx = ExtractAttributesContext {
            previous_episodes: &prev,
            episode_content: &episode,
            node: &node,
        };
        let msgs = extract_attributes(&ctx);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].role, Role::User);
        assert!(msgs[0].content.ends_with(DO_NOT_ESCAPE_UNICODE));
        assert!(
            msgs[0]
                .content
                .contains("You are an entity attribute extraction specialist.")
        );
        let u = &msgs[1].content;
        assert!(
            u.starts_with("Given the MESSAGES and the following ENTITY, update its attributes.")
        );
        assert!(u.contains("HARD RULES — violating any of these is a failure:"));
        assert!(u.contains(
            "6. NEVER infer attribute values from the entity's name, from related entities, from"
        ));
        // node content interpolated as JSON with Python-default separators (", " / ": ")
        assert!(u.contains("{\"name\": \"Mary\"}"));
    }
}
